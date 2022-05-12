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

from multiprocessing import Pipe
from environmental import EnvironmentalHandler

import time

import rospy

from utils.msg import environmental

class environmentalNODE():
    
    def __init__(self):
        beacon = 23456
        id = 106
        serverpublickey = '/home/pi/dei_ws/src/output/src/environmentalserver/publickey_server.pem'
        clientprivatekey = '/home/pi/dei_ws/src/output/src/environmentalserver/privatekey_client.pem'
        
        gpsStR, self.gpsStS = Pipe(duplex = False)

        self.envhandler = EnvironmentalHandler(id, beacon, serverpublickey, gpsStR, clientprivatekey)
    
        rospy.init_node('environmentalNODE', anonymous=False)
        
        # Environmental subscriber object
        self.ENVIRONMENTAL_subscriber = rospy.Subscriber("/automobile/environment", environmental, self._send)
        
        
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start subscriber.
        """
        rospy.loginfo("starting environmentalNODE")
        self.envhandler.start()
        time.sleep(5)
        rospy.spin()        

    def _send(self, msg):
        a = {"obstacle_id": msg.obstacle_id, "x": msg.x, "y": msg.y}
        self.gpsStS.send(a)

if __name__ == '__main__':
    envNOD = environmentalNODE()
    envNOD.run()

    envNOD.envhandler.stop()
    envNOD.envhandler.join()
