#pragma once

#include "STrack.h"

class BYTETracker
{
public:
	BYTETracker(string &model_path, int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	void update(string &video_path);

private:
	Size get_size(int vw, int vh, int dw, int dh);
	Mat letterbox(Mat img, int height, int width);
	torch::Tensor non_max_suppression(torch::Tensor prediction);
	torch::Tensor xywh2xyxy(torch::Tensor x);
	torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores, float overlap);
	void scale_coords(torch::Tensor &coords, Size img_size, Size img0_shape);

private:
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void embedding_distance(vector<STrack*> &tracks, vector<STrack> &detections,
		vector<vector<float> > &cost_matrix, int *cost_matrix_size, int *cost_matrix_size_size);
	void fuse_motion(vector<vector<float> > &cost_matrix, vector<STrack*> &tracks, vector<STrack> &detections, 
		bool only_position = false, float lambda_ = 0.98);
	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

	Scalar get_color(int idx);

private:
	torch::jit::script::Module jde_model;
	torch::Device *device;

	int net_width;
	int net_height;
	float conf_thresh;
	float nms_thresh;
	int frame_id;
	int max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};