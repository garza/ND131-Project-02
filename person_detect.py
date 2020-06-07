%%writefile person_detect.py

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d

class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.plugin = None
        self.model = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_blob_name = None
        self.output_blob = None
        self.ie = IECore()

        try:
            ##log.info("Loading network files:\n\t{}\n\t{}".format(self.model_structure, self.model_weights))
            self.network = self.ie.read_network(self.model_structure, self.model_weights)
            versions = self.ie.get_versions(self.device)
            ##log.info("{}{}".format(" "*8, self.device))
            ##log.info("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[self.device].major, versions[self.device].minor))
            ##log.info("{}Build ........... {}".format(" "*8, versions[self.device].build_number))

        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.network.inputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_name=next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_name].shape
        ##log.info("network output name")
        ##log.info(self.output_name)
        ##log.info("network output info")
        ##log.info(self.output_shape)
        ##log.info("network input shape")
        ##log.info(self.input_name)
        ##log.info(self.input_shape)

    def load_model(self):
        self.exec_network = self.ie.load_network(self.network, self.device)
        ## grab the input layer ##
        self.input_blob = self.network.inputs
        ##next(iter(self.network.inputs))
        img_info_input_blob = None
        self.feed_dict = {}
        for blob_name in self.network.inputs:
            ##log.info("blob_name: " + blob_name)
            if len(self.network.inputs[blob_name].shape) == 4:
                self.input_blob_name = blob_name
            elif len(self.network.inputs[blob_name].shape) == 2:
                img_info_input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(net.inputs[blob_name].shape), blob_name))
        self.output_blob = next(iter(self.network.outputs))

    def predict(self, image):
        input_image = self.preprocess_input(image)
        input_dict = {self.input_name: input_image}
        results = self.exec_network.infer(inputs=input_dict)
        coords = self.preprocess_outputs(results, image)
        ##log.info("coords:")
        ##log.info(coords)
        self.draw_outputs(coords, image)
        return coords, image

    def draw_outputs(self, coords, image):
        for idx, box in enumerate(coords):
            color = (int(min((idx + 80) * 12.5, 255)),
                 min(box[5] * 7, 255), min(box[5] * 5, 255))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

    def preprocess_outputs(self, outputs, image):
        width = image.shape[1]
        height = image.shape[0]
        coords = []
        detection_out_results = outputs[self.output_name][0][0]
        for boxid, box in enumerate(detection_out_results):
            if (box[2] > self.threshold):
                ## output format:  [image_id, label, conf, x_min, y_min, x_max, y_max]
                label = box[1]
                conf = box[2]
                xmin = np.int(width * box[3])
                ymin = np.int(height * box[4])
                xmax = np.int(width * box[5])
                ymax = np.int(height * box[6])
                coords.append([xmin, ymin, xmax, ymax, conf, label])

        return coords

    def preprocess_input(self, image):
        image=cv2.resize(image, (544, 320), interpolation = cv2.INTER_AREA)
        image=np.moveaxis(image, -1, 0)
        return image

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()

    try:
        ##log.info("loading queue")
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        ##log.info("attempting to open video file: %s", video_file)
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_output_path = os.path.join(output_path, 'output_video.mp4')
    ##log.info("using output path %s", video_output_path)
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()
    ##log.info("start time is %d", start_inference_time)

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            #log.info("counter at %d", counter)
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25

            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)

        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    args=parser.parse_args()

    main(args)
