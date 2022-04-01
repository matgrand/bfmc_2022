/**
 * Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/


#include <Message.hpp>

/*
* This class handles creates the message to send over serial based on the command and on it's value.
*/

/**
 * @name    getTextForKey
 * @brief   
 *
 * Provide the cmd key associated to an action.
 *
 * @param [in] enumVal     The integer value corresponding to the action, in the action enumeration
 *                         (zero-indexed).
 *
 * @retval String associated to the action.
 *
 * Example Usage:
 * @code
 *    getTextForKey(1);
 * @endcode
 */
std::string message::getTextForKey(int enumVal){
    return ActionStrings[enumVal];
}

/**
 * @name    speed
 * @brief   
 *
 * Construct the string to be sent, associated to speed action.
 *
 * @param [in] f_velocity  Velocity.
 *
 * @retval Complete string for send command.
 *
 * Example Usage:
 * @code
 *    speed(1.234);
 * @endcode
 */
std::string message::speed(float f_velocity){
    std::stringstream strs;
    char buff[100];
    snprintf(buff, sizeof(buff),"%.2f;;\r\n",f_velocity);
    strs<<"#"<<getTextForKey(0)<<":"<<buff;
    return strs.str();
}

/**
 * @name    steering
 * @brief   
 *
 * Construct the string to be sent, associated to steering action.
 *
 * @param [in] f_angle     Angle.
 *
 * @retval Complete string for send command.
 *
 * Example Usage:
 * @code
 *    steering(5.678);
 * @endcode
 */
std::string message::steer(float f_angle){
    std::stringstream strs;
    char buff[100];
    snprintf(buff, sizeof(buff),"%.2f;;\r\n",f_angle);
    strs<<"#"<<getTextForKey(1)<<":"<<buff;
    return strs.str();
}

/**
 * @name    brake
 * @brief   
 *
 * Provide the cmd key associated to an action.
 *
 * @param [in] f_angle     Angle.
 *
 * Construct the string to be sent, associated to an action.
 *
 * Example Usage:
 * @code
 *    brake(1.234);
 * @endcode
 */
std::string message::brake(float f_angle){
    std::stringstream strs;
    char buff[100];
    snprintf(buff, sizeof(buff),"%.2f;;\r\n",f_angle);
    strs<<"#"<<getTextForKey(2)<<":"<<buff;
    return strs.str();
}

/**
 * @name    pida
 * @brief   
 *
 * Construct the string to be sent, associated to pid activating.
 *
 * @param [in] activate     Set PID active or not.
 *
 * @retval Complete string for send command.
 *
 * Example Usage:
 * @code
 *    pida(true);
 * @endcode
 */
std::string message::pida(bool activate){
    std::stringstream strs;
    char buff[100];
    snprintf(buff, sizeof(buff),"%d;;\r\n",activate);
    strs<<"#"<<getTextForKey(3)<<":"<<buff;
    return strs.str();
}

/**
 * @name    enpb
 * @brief   
 *
 * Construct the string to be sent, associated to encoder publisher activating.
 *
 * @param [in] activate     Set ENPB active or not.
 *
 * @retval Complete string for send command.
 *
 * Example Usage:
 * @code
 *    enpb(true);
 * @endcode
 */
std::string message::enpb(bool activate){
    std::stringstream strs;
    char buff[100];
    snprintf(buff, sizeof(buff),"%d;;\r\n",activate);
    strs<<"#"<<getTextForKey(4)<<":"<<buff;
    return strs.str();
}


/**
 * @name    pids
 * @brief   
 *
 * Construct the string to be sent, associated to pid settig.
 *
 * @param [in] kp          Param kp.
 * @param [in] ki          Param ki.
 * @param [in] kd          Param kd.
 * @param [in] tf          Param tf.
 *
 * @retval Complete string for send command.
 *
 * Example Usage:
 * @code
 *    pids(1.234567,7.654321,12.12121212,3.4567890);
 * @endcode
 */
std::string message::pids(float kp,float ki,float kd,float tf){
    std::stringstream strs;
    char buff[100];
    snprintf(buff, sizeof(buff),"%.5f;%.5f;%.5f;%.5f;;\r\n",kp,ki,kd,tf);
    strs<<"#"<<getTextForKey(5)<<":"<<buff;
    return strs.str();
}
