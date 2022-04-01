#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
        
import io
import numpy as np
import time

import rospy

from cv_bridge       import CvBridge
from sensor_msgs.msg import Image

class cameraSpooferNODE():
    def __init__(self, ext = '.h264'):
        """Node used for spoofing a camera/ publishing a video stream from a folder 
        with videos
        """
        # params
        self.videoSize = (640,480)
        
        self.videoDir = "path/to/videos/directory"
        self.videos = self.open_files(self.videoDir, ext = ext)
        
        rospy.init_node('cameraSpooferNODE', anonymous=False)
        self.image_publisher = rospy.Publisher("/automobile/image_raw", Image, queue_size=1)
        
        self.bridge = CvBridge()
        
    #================================ RUN ================================================
    def run(self):
        """Apply the initializing methods and start the thread. 
        """
        rospy.loginfo("starting camaeraSpooferNODE")
        self._play_video(self.videos)
      
    # ===================================== INIT VIDEOS ==================================
    def open_files(self, inputDir, ext):
        """Open all files with the given path and extension
        """
        
        files =  glob.glob(inputDir + '/*' + ext)  
        return files

    # ===================================== PLAY VIDEO ===================================
    def _play_video(self, videos):
        """Iterate through each video in the folder, open a cap and publish the frames.
        """
        while True:
            for video in videos:
                cap         =   cv2.VideoCapture(video)
                
                while True:
                    ret, frame = cap.read()
                    stamp = time.time()
                    if ret: 
                        frame = cv2.resize(frame, self.videoSize)
                        # output image and time stamp
                        # Note: The sending process can be blocked, when doesn't exist any consumer process and it reaches the limit size.
                        try:
                            imageObject = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                            imageObject.header.stamp = rospy.Time.now()
                            self.image_publisher.publish(imageObject)
                        except CvBridgeError as e:
                            print(e)
                               
                    else:
                        break

                cap.release()
            
if __name__ == "__main__":
    camNod = cameraSpooferNODE()
    camNod.run()
