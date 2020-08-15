# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import cv2
import datetime
import glob
import json
from json import JSONEncoder
import multiprocessing as mp
import numpy
import os
import tqdm
from detectron2.data import MetadataCatalog

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor_fade import VisualizationDemo

VERSION = '0.9'


class custom_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        else:
            return super(custom_encoder, self).default(obj)


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
        description="Detectron2-based Video Ingest System for FADE for built-in models")
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


def scale_box(box):
    bx = box.numpy().tolist()
    bx[0] = bx[0]/width
    bx[1] = bx[1]/height
    bx[2] = bx[2]/width
    bx[3] = bx[3]/height
    return bx


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

        segments = []
        
        for instance, vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            output_file.write(vis_frame)
            segments.append(instance)
        video.release()

        if args.output:
            with open(output_fname + ".json", 'w') as segments_file:
                for i,instance in enumerate(segments):
                    obj = {}

                    # only include header in first row (to avoid having to keep everything in memory before writing)
                    if i == 0:
                        obj['version'] = VERSION
                        obj['date'] = datetime.datetime.now()
                        obj['source_file'] = args.video_input
                        obj['number_frames'] = len(segments)
                        obj['fps'] = frames_per_second
                        obj['width'] = width
                        obj['height'] = height

                    obj['t'] = i/frames_per_second
                    obj['objects'] = []

                    to_cpu = instance.to('cpu')

                    classes = to_cpu.pred_classes
                    scores = to_cpu.scores
                    boxes = to_cpu.pred_boxes.tensor

                    for j in range(0,len(classes)):
                        obj['objects'].append({
                            'class': thing_classes[classes[j]],
                            'score': scores[j].numpy(),
                            'box': scale_box(boxes[j,:])
                        })

                    #print(json.dumps(obj, cls=custom_encoder))
                    json.dump(obj, segments_file, indent=2, cls=custom_encoder)
            output_file.release()
