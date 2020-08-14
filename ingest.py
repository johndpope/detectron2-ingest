# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import cv2
import glob
import json
from json import JSONEncoder
import multiprocessing as mp
import numpy
import os
import time
import tqdm
from detectron2.data import MetadataCatalog

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor_fade import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

########
# Argh
# https://pynative.com/python-serialize-numpy-ndarray-into-json/
##


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
########


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2-based Video Ingest System for FADE for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A directory to save outputs. ",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    thing_classes = metadata.thing_classes

    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            #assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
        segments_data = {
            'info': {'description': "Fade segmentation", "version": "0.9"}}
        segments_data['categories'] = {}
        segments_data['annotations'] = []
        for instance, vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            # if args.output:
            output_file.write(vis_frame)
            segments_data['annotations'].append(instance)
            # else:
            #     cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            #     cv2.imshow(basename, vis_frame)
            #     if cv2.waitKey(1) == 27:
            #         break  # esc to quit
        video.release()
        if args.output:
            with open(output_fname+".json", 'w') as segments_file:
                for i,instance in enumerate(segments_data['annotations']):
                    obj = { "t": (i/frames_per_second), 'objects': [] } 
                    to_cpu = instance.to('cpu')

                    pred_classes = to_cpu.pred_classes
                    scores = to_cpu.scores
                    pred_boxes = to_cpu.pred_boxes.tensor

                    for j in range(0,len(pred_classes)):
                        obj['objects'].append({
                            'class': thing_classes[pred_classes[j]],
                            'score': scores[j].numpy(),
                            'box': pred_boxes[j,:].numpy().tolist()
                        })
                        
                    print(json.dumps(obj, cls=NumpyArrayEncoder))
                    json.dump(obj, segments_file, indent=2, cls=NumpyArrayEncoder)
            output_file.release()
