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

from utils.msg import vehicles


class vehicletovehicleNODE():
    
    ## Constructor.
    #  @param self          The object pointer.
    def __init__(self):
        rospy.init_node('vehicletovehicleNODE', anonymous=False)
        
        # vehicles publisher objects
        self.Vehicles_publisher = rospy.Publisher("/automobile/vehicles", vehicles, queue_size=1)
        
    #================================ RUN ========================================  
    def run(self):
        """ Method for running listener algorithm.
        """
        rospy.loginfo("starting vehicletovehicleNODE")
        self._init_socket()
        self._getting()
                
     #================================ INIT SOCKET ========================================
    def _init_socket(self):
        self.PORT = 50009
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #(internet, UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind(('', self.PORT))
        self.sock.settimeout(1)
    
    #================================ GETTING ========================================
    def _getting(self): 
        # Listen for incomming broadcast messages
        while not rospy.is_shutdown():
            try:
                data,_ = self.sock.recvfrom(4096) # buffer size is 1024 bytes
                # Decode received data
                data = data.decode("utf-8") 
                data = json.loads(data)
                
                ID = int(data['id'])
                timestamp = float(data['timestamp'])
                pos = complex(data['coor'])
                ang = complex(data['rot'])
                
                veh = vehicles()
                veh.ID = ID
                veh.state = state
                veh.posA = pos['real']
                veh.posB = pos['imag']
                veh.rotA = rot['real']
                veh.rotB = rot['imag']
                self.Vehicles_publisher(veh)
                
            except Exception as e:
                print("Receiving data failed with error: " + str(e))

if __name__ == "__main__":
    vehNOD = vehicletovehicleNODE()
    vehNOD.run()