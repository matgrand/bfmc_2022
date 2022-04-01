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

import socket
import struct
import time
import cv2
        
import rospy

from cv_bridge       import CvBridge
from sensor_msgs.msg import Image

class cameraDealerNODE():
    def __init__(self):
        """Process used for sending images over the network to a targeted IP via UDP protocol 
        (no feedback required). The image is compressed before sending it. 

        Used for visualizing your raspicam images on remote PC. 
        """
        
        rospy.init_node('cameraDealerNODE', anonymous=False)
        
        # Image publisher object
        rospy.Subscriber("/automobile/image_raw", Image, self._streams)
        
        self.connection = None
        
        self.bridge = CvBridge()
        
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start subscriber.
        """
        rospy.loginfo("starting cameraDealerNODE")
        self._init_socket()
        
        rospy.spin()
    
     # ===================================== INIT SOCKET ==================================
    def _init_socket(self):
        """Initialize the socket client. 
        """
        serverIp   =  '192.168.88.201' # PC ip
        port       =  2244            # com port
        
        self.client_socket = socket.socket()
        self.connection = None
        # Trying repeatedly to connect the camera receiver.
        while self.connection is None:
            try:
                self.client_socket.connect((serverIp, port))
                self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                self.connection = self.client_socket.makefile('wb') 
            except Exception as e:
                time.sleep(0.5)
                pass

    # ===================================== SEND ==================================
    def _streams(self, msg):
        """Sending the frames received from the topic to the remote client by using the created socket connection. 
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        
        if self.connection is not None:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                result, image = cv2.imencode('.jpg', image, encode_param)
                data   =  image.tobytes()
                size   =  len(data)

                self.connection.write(struct.pack("<L",size))
                self.connection.write(data)
            except Exception as e:
                #rospy.loginfo("CameraDealerNode failed to streamer images:",e,"\n")
                # Reinitialize the socket for reconnecting to client.  
                self.connection = None
                self._init_socket()
                pass
            
if __name__ == "__main__":
    camDealerNod = cameraDealerNODE()
    camDealerNod.run()
