
/**
 * message.hpp - Header file
 * message converter functions declarations
 */

#ifndef _MESSAGE_HPP_
#define _MESSAGE_HPP_

#include <string>
#include <sstream>
#include <complex>
#include <stdio.h>

namespace message{

    //enum for defining the actions that can be performed 
    // 1 - SPEED Command
    // 2 - STEERING Command
    // 3 - BRAKE Command
    // 4 - PID ACTIVATION Command
    // 5 - ENCODER PUBLISHER Command
    // 6 - PID TUNNING Command
    // 7 - NO Command

    //the strings associated to each action
    static std::string ActionStrings[] = { "1", "2", "3" , "4" , "5" , "6", "7"};

    std::string getTextForKey(int);
    std::string speed(float);
    std::string steer(float);
    std::string brake(float);
    std::string pida(bool);
    std::string enpb(bool);
    std::string pids(float,float,float,float);
};

#endif
