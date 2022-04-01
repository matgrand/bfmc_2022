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
import json

import rospy

from std_msgs.msg import Byte

class semaphoreNODE():
    def __init__(self):
        """listener class. 
        
        Class used for running port listener algorithm 
        """                
        rospy.init_node('semaphoreNODE', anonymous=False)
        
        # BNO publisher object
        self.Semaphoremaster_publisher = rospy.Publisher("/automobile/semaphore/master", Byte, queue_size=1)
        self.Semaphoreslave_publisher = rospy.Publisher("/automobile/semaphore/slave", Byte, queue_size=1)
        self.Semaphoreantimaster_publisher = rospy.Publisher("/automobile/semaphore/antimaster", Byte, queue_size=1)
        self.Semaphorestart_publisher = rospy.Publisher("/automobile/semaphore/start", Byte, queue_size=1)

    #================================ RUN ========================================
    def run(self):
        """ Method for running listener algorithm.
        """
        rospy.loginfo("starting semaphoreNODE")
        self._init_socket()
        self._getting()
        
    #================================ INIT SOCKET ========================================
    def _init_socket(self):
        # Communication parameters, create and bind socket
        self.PORT = 50007
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #(internet, UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind(('', self.PORT))
        self.sock.settimeout(1)
        
        
    #================================ GETTING ========================================
    def _getting(self): 
        # Listen for incomming broadcast messages
        while not rospy.is_shutdown():
            # Wait for data
            try:
                data, addr = self.sock.recvfrom(4096) # buffer size is 1024 bytes
                dat = data.decode('utf-8')
                dat = json.loads(dat)
                ID = int(dat['id'])
                state = int(dat['state'])
                if (ID == 1):
                    self.Semaphoremaster_publisher(state)
                elif (ID == 2):
                    self.Semaphoreslave_publisher(state)
                elif (ID == 3):
                    self.Semaphoreantimaster_publisher(state)
                elif (ID == 4):
                    self.Semaphorestart_publisher(state)

            except Exception as e:
                if str(e) !="timed out":
                    print("Receiving data failed with error: " + str(e))
                
if __name__ == "__main__":
    semNOD = semaphoreNODE()
    semNOD.run()