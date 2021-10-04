/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(vector<float> tlwh_, float score, vector<float> temp_feat, int buffer_size=30);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id, bool update_feature = true);

public:
	bool is_activated;
	int track_id;
	int state;
	vector<float> curr_feat;
	vector<float> smooth_feat;
	float alpha;

	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;

private:
	void update_features(vector<float> temp_feat);
	byte_kalman::KalmanFilter kalman_filter;
};