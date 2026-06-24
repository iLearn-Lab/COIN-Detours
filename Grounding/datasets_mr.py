import os
import json
from PIL import Image
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd

class VideoCentricDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self,
        data_path: str,
        video_folder: Optional[str] = None,
        feat_folder: Optional[str] = None,
        fps: int = 2,
        split='train',
        num_clips=32,
        clip_length=-1,
        video1time=15,
    ) -> None:
        super(VideoCentricDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.video_folder = video_folder
        self.feat_folder = feat_folder
        self.fps = fps
        self.is_text_only = [
            False
            for source in self.list_data_dict
        ]
        self.split = split
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.video1time = video1time

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def construct_messages_mr_fps(
        self,
        video2_path,
        feature2_path,
        video1_path,
        feature1_path,
        fps,
        querys,
        temporal_windows,
        retrieval_segment,
        retrieval_mode,
        video1_end,
        video1time,
    ):
        unified_instruction_mr_seg = (
            "This is a sequence interleaved with timestamps and frames. "
            "Your task is to identify the specific timestamp(s) based on the user's question, "
            "which is derived from a background video. "

            "Directly output the timestamp(s) that best answer the user's question."
        )

        unified_instruction_mr = (
            "This is a sequence interleaved with timestamps and frames. "
            "Your task is to identify the temporal window (start and end timestamps) "
            "based on the user's question, which is derived from a background video. "

            "Directly output the start and end timestamps that best answer the user's question."
        )

        video1_start = 0 if video1time == -1 else max(0, video1_end - video1time)

        if retrieval_mode == 'mr_seg':
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"{video2_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1],
                            "feature": f"{feature2_path}", "num_clips": self.num_clips, "clip_length": self.clip_length, "temporal_windows":temporal_windows},
                        {"type": "text", "text": f"{unified_instruction_mr_seg} Next is the background video."},
                        {"type": "video", "video": f"{video1_path}", "fps": 1, "video_start": video1_start, "video_end": video1_end,"feature": f"{feature1_path}",
                        "num_clips": self.num_clips, "clip_length": self.clip_length,},

                    ]
                },
            ]
        elif retrieval_mode == 'mr':
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"{video2_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
                        {"type": "text", "text": f"{unified_instruction_mr} Next is the background video."},
                        {"type": "video", "video": f"{video1_path}", "fps": 1, "video_start": video1_start, "video_end": video1_end,"feature": f"{feature1_path}",
                        "num_clips": self.num_clips, "clip_length": self.clip_length,}
                    ]
                },
            ]

        for query in querys:
            message.append(
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"User question:{query}\nAnswer: "}
                ]
                }
            )

        return message


    def __getitem__(self, i) -> Dict[str, List]:


        source = self.list_data_dict[i]
        qid = source["qid"]
        vid = source["id"]
        annos = source["annos"]
        retrieval_mode = source["mode"]

        video_start = source.get("video_start", 0)
        video_end = source.get("video_end", source["duration"])

        video1_end = source["video1_startends"][0]

        temporal_window = [anno["window"] for anno in annos]
        query = [anno["query"] for anno in annos]
        duration = video_end - video_start

        retrieval_segment = [video_start, video_end]

        video2_path = source.get("video2_path", None)
        video1_path = source.get("video1_path", None)

        feature1_path = None
        feature2_path = None

        message = self.construct_messages_mr_fps(video2_path=video2_path, feature2_path=feature2_path, video1_path=video1_path, feature1_path=feature1_path, fps=self.fps, querys=query, temporal_windows=temporal_window,
                                                 retrieval_segment=retrieval_segment, retrieval_mode=retrieval_mode, video1_end=video1_end, video1time=self.video1time)

        return {"message":message, "split":self.split, "temporal_window":temporal_window, "mode":retrieval_mode, "qid":qid, "duration":duration}