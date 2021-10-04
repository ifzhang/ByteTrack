#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(string &model_path, int frame_rate, int track_buffer)
{
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		cout << "CUDA available! Test on GPU." << endl;
		device_type = torch::kCUDA;
	}
	else {
		cout << "Test on CPU." << endl;
		device_type = torch::kCPU;
	}

	torch::set_num_threads(1);
	cout << "set threads num: " << torch::get_num_threads() << endl;

	net_width = 576;
	net_height = 320;
	conf_thresh = 0.5;
	nms_thresh = 0.4;
	device = new torch::Device(device_type);

	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer);

	cout << "Load model ... ";
	jde_model = torch::jit::load(model_path);
	jde_model.to(*device);
	cout << "Done!" << endl;
}

BYTETracker::~BYTETracker()
{
	delete device;
}

void BYTETracker::update(string &video_path)
{
	VideoCapture cap(video_path);
	if (!cap.isOpened())
		return;

	int vw = cap.get(CAP_PROP_FRAME_WIDTH);
	int vh = cap.get(CAP_PROP_FRAME_HEIGHT);
	Size size = get_size(vw, vh, net_width, net_height);

	Mat img0;
	while (true)
	{
		cap >> img0;
		if (img0.empty())
			break;

		resize(img0, img0, size);
		Mat img = letterbox(img0, net_height, net_width);

		Mat img_rgb;
		cvtColor(img, img_rgb, COLOR_BGR2RGB);
		Mat img_float;
		img_rgb.convertTo(img_float, CV_32FC3);
		img_float /= 255.0;

		auto img_tensor = torch::from_blob(img_float.data, { net_height, net_width, 3 }, torch::kFloat32);
		auto img_tensor_unsqueeze = torch::unsqueeze(img_tensor, 0);
		img_tensor_unsqueeze = img_tensor_unsqueeze.permute({ 0, 3, 1, 2 });

		// Create a vector of inputs.
		vector<torch::jit::IValue> inputs;
		inputs.push_back(img_tensor_unsqueeze.to(*device));

		////////////////// Step 1: Network forward, get detections & embeddings //////////////////
		this->frame_id++;
		vector<STrack> activated_stracks;
		vector<STrack> refind_stracks;
		vector<STrack> removed_stracks;
		vector<STrack> lost_stracks;
		vector<STrack> detections;

		vector<STrack> detections_cp;
		vector<STrack> tracked_stracks_swap;
		vector<STrack> resa, resb;
		vector<STrack> output_stracks;

		vector<STrack*> unconfirmed;
		vector<STrack*> tracked_stracks;
		vector<STrack*> strack_pool;
		vector<STrack*> r_tracked_stracks;

		torch::Tensor pred_gpu = jde_model.forward(inputs).toTensor();
		auto pred_cpu = pred_gpu.to(torch::kCPU).squeeze(0);
		auto pred_thresh = pred_cpu.index_select(0, torch::nonzero(pred_cpu.select(1, 4) > this->conf_thresh).squeeze());

		if (pred_thresh.sizes()[0] > 0)
		{
			auto dets = non_max_suppression(pred_thresh);
			scale_coords(dets.slice(1, 0, 4), Size(this->net_width, this->net_height), Size(img0.cols, img0.rows));
			// Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)
			for (int i = 0; i < dets.sizes()[0]; i++)
			{
				vector<float> tlbr_;
				tlbr_.resize(4);
				tlbr_[0] = dets[i][0].item<float>();
				tlbr_[1] = dets[i][1].item<float>();
				tlbr_[2] = dets[i][2].item<float>();
				tlbr_[3] = dets[i][3].item<float>();

				//rectangle(img0, Rect(Point(tlbr_[0], tlbr_[1]), Point(tlbr_[2], tlbr_[3])), Scalar(0, 255, 0), 2);

				float score = dets[i][4].item<float>();

				vector<float> temp_feat;
				for (int j = 6; j < dets.sizes()[1]; j++)
				{
					temp_feat.push_back(dets[i][j].item<float>());
				}

				STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, temp_feat, 30);
				detections.push_back(strack);
			}
		}

		// Add newly detected tracklets to tracked_stracks
		for (int i = 0; i < this->tracked_stracks.size(); i++)
		{
			if (!this->tracked_stracks[i].is_activated)
				unconfirmed.push_back(&this->tracked_stracks[i]);
			else
				tracked_stracks.push_back(&this->tracked_stracks[i]);
		}

		////////////////// Step 2: First association, with embedding //////////////////
		strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
		STrack::multi_predict(strack_pool, this->kalman_filter);

		vector<vector<float> > dists;
		int dist_size = 0, dist_size_size = 0;
		embedding_distance(strack_pool, detections, dists, &dist_size, &dist_size_size);
		fuse_motion(dists, strack_pool, detections);

		vector<vector<int> > matches;
		vector<int> u_track, u_detection;
		linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_track, u_detection);

		for (int i = 0; i < matches.size(); i++)
		{
			STrack *track = strack_pool[matches[i][0]];
			STrack *det = &detections[matches[i][1]];
			if (track->state == TrackState::Tracked)
			{
				track->update(*det, this->frame_id);
				activated_stracks.push_back(*track);
			}
			else
			{
				track->re_activate(*det, this->frame_id, false);
				refind_stracks.push_back(*track);
			}
		}

		////////////////// Step 3: Second association, with IOU //////////////////
		for (int i = 0; i < u_detection.size(); i++)
		{
			detections_cp.push_back(detections[u_detection[i]]);
		}
		detections.clear();
		detections.assign(detections_cp.begin(), detections_cp.end());
		
		for (int i = 0; i < u_track.size(); i++)
		{
			if (strack_pool[u_track[i]]->state == TrackState::Tracked)
			{
				r_tracked_stracks.push_back(strack_pool[u_track[i]]);
			}
		}

		dists.clear();
		dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

		matches.clear();
		u_track.clear();
		u_detection.clear();
		linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_track, u_detection);

		for (int i = 0; i < matches.size(); i++)
		{
			STrack *track = r_tracked_stracks[matches[i][0]];
			STrack *det = &detections[matches[i][1]];
			if (track->state == TrackState::Tracked)
			{
				track->update(*det, this->frame_id);
				activated_stracks.push_back(*track);
			}
			else
			{
				track->re_activate(*det, this->frame_id, false);
				refind_stracks.push_back(*track);
			}
		}

		for (int i = 0; i < u_track.size(); i++)
		{
			STrack *track = r_tracked_stracks[u_track[i]];
			if (track->state != TrackState::Lost)
			{
				track->mark_lost();
				lost_stracks.push_back(*track);
			}
		}

		// Deal with unconfirmed tracks, usually tracks with only one beginning frame
		detections_cp.clear();
		for (int i = 0; i < u_detection.size(); i++)
		{
			detections_cp.push_back(detections[u_detection[i]]);
		}
		detections.clear();
		detections.assign(detections_cp.begin(), detections_cp.end());

		dists.clear();
		dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

		matches.clear();
		vector<int> u_unconfirmed;
		u_detection.clear();
		linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

		for (int i = 0; i < matches.size(); i++)
		{
			unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
			activated_stracks.push_back(*unconfirmed[matches[i][0]]);
		}

		for (int i = 0; i < u_unconfirmed.size(); i++)
		{
			STrack *track = unconfirmed[u_unconfirmed[i]];
			track->mark_removed();
			removed_stracks.push_back(*track);
		}

		////////////////// Step 4: Init new stracks //////////////////
		for (int i = 0; i < u_detection.size(); i++)
		{
			STrack *track = &detections[u_detection[i]];
			if (track->score < this->conf_thresh)
				continue;
			track->activate(this->kalman_filter, this->frame_id);
			activated_stracks.push_back(*track);
		}

		////////////////// Step 5: Update state //////////////////
		for (int i = 0; i < this->lost_stracks.size(); i++)
		{
			if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
			{
				this->lost_stracks[i].mark_removed();
				removed_stracks.push_back(this->lost_stracks[i]);
			}
		}
		
		for (int i = 0; i < this->tracked_stracks.size(); i++)
		{
			if (this->tracked_stracks[i].state == TrackState::Tracked)
			{
				tracked_stracks_swap.push_back(this->tracked_stracks[i]);
			}
		}
		this->tracked_stracks.clear();
		this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

		this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
		this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

		this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
		for (int i = 0; i < lost_stracks.size(); i++)
		{
			this->lost_stracks.push_back(lost_stracks[i]);
		}

		this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
		for (int i = 0; i < removed_stracks.size(); i++)
		{
			this->removed_stracks.push_back(removed_stracks[i]);
		}

		remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

		this->tracked_stracks.clear();
		this->tracked_stracks.assign(resa.begin(), resa.end());
		this->lost_stracks.clear();
		this->lost_stracks.assign(resb.begin(), resb.end());
		
		for (int i = 0; i < this->tracked_stracks.size(); i++)
		{
			if (this->tracked_stracks[i].is_activated)
			{
				output_stracks.push_back(this->tracked_stracks[i]);
			}
		}

		for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 200 && !vertical)
			{
				Scalar s = get_color(output_stracks[i].track_id);
				rectangle(img0, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
				//putText(img0, to_string(output_stracks[i].track_id), Point(tlwh[0], tlwh[1]), 0, 0.6, s, 2);
			}
		}

		imshow("test", img0);
		if (waitKey(1) > 0)
			break;
	}
	cap.release();
}