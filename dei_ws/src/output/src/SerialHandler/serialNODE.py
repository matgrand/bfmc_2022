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

import serial
from filehandler      import FileHandler
from messageconverter import MessageConverter
import json

import rospy

from std_msgs.msg      import String
from utils.srv        import subscribing, subscribingResponse

class serialNODE():
    def __init__(self):
        """It forwards the control messages received from socket to the serial handling node. 
        """
        devFile = '/dev/ttyACM0'
        logFile = 'historyFile.txt'
        
        # comm init       
        self.serialCom = serial.Serial(devFile,19200,timeout=0.1)
        self.serialCom.flushInput()
        self.serialCom.flushOutput()

        # log file init
        self.historyFile = FileHandler(logFile)
        
        # message converted init
        self.messageConverter = MessageConverter()
        
        self.buff=""
        self.isResponse=False
        self.__subscribers={}
        
        rospy.init_node('serialNODE', anonymous=False)
        
        self.command_subscriber = rospy.Subscriber("/automobile/command", String, self._write)
        
        self.subscribe = rospy.Service("command_feedback_en", subscribing, self._subscribe)        
    
     # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads
        """
        rospy.loginfo("starting serialNODE")
        self._read()    
        
    # ===================================== READ ==========================================
    def _read(self):
        """ It's represent the reading activity on the the serial.
        """
        while not rospy.is_shutdown():
            read_chr=self.serialCom.read()
            try:
                read_chr=(read_chr.decode("ascii"))
                if read_chr=='@':# begin of message from serial
                    self.isResponse=True
                    if len(self.buff)!=0:
                        self.__checkSubscriber(self.buff)
                    self.buff=""
                elif read_chr=='\r':# end of message from serial   
                    self.isResponse=False
                    if len(self.buff)!=0:
                        # print(self.buff)
                        self.__checkSubscriber(self.buff)
                    self.buff=""
                if self.isResponse:
                    self.buff+=read_chr
                self.historyFile.write(read_chr)
                 
            except UnicodeDecodeError:
                pass
    
    def __checkSubscriber(self,f_response):
        """ Checking the list of the waiting object to redirectionate the message to them. 
        """
        l_key=f_response[1:5]
        if l_key in self.__subscribers:
            subscribers = self.__subscribers[l_key]
            for sub in subscribers:
                sub.publish(f_response)
    
    def _subscribe(self, msg):
        """Enable the feedback from from a type of command to ensure receiving from embedded.
        To avoid topic names collision, specify the node name and code in the topic itself. 
        """
        if msg.subscribing:
            if msg.code in self.__subscribers:
                for sub in self.__subscribers[msg.code]:
                    if (msg.topic == sub.name):
                        return subscribingResponse(False)
                    else:
                        command_publisher = rospy.Publisher(msg.topic, String, queue_size=1)
                        self.__subscribers[msg.code].append(command_publisher)
                        return subscribingResponse(True)
            else:
                command_publisher = rospy.Publisher(msg.topic, String, queue_size=1)
                self.__subscribers[msg.code] = [command_publisher]
                return subscribingResponse(True)
        else:
            if msg.code in self.__subscribers:     
                for sub in self.__subscribers[msg.code]:
                    if (msg.topic == sub.name):
                        sub.unregister()
                        self.__subscribers[msg.code].remove(sub)
                        subscribingResponse(True)
                    else:
                        return subscribingResponse(False)
            else:
                raise subscribingResponse(False)
        
        
    # ===================================== WRITE ==========================================
    def _write(self, msg):
        """ Represents the writing activity on the the serial.
        """
        command = json.loads(msg.data)
        # Unpacking the dictionary into action and values
        command_msg = self.messageConverter.get_command(**command)
        self.serialCom.write(command_msg.encode('ascii'))
        self.historyFile.write(command_msg)
            
            
if __name__ == "__main__":
    serNod = serialNODE()
    serNod.run()
