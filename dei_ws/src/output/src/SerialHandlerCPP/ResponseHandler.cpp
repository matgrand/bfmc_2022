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


#include <ResponseHandler.hpp>

/*
* This class handles the receiving of the messages from the Embedded via serial. 
*/


void BaseResponseHandler::deactive(){
    m_active=false;
}

ResponseHandler::ResponseHandler()
                                :m_isResponse(false)
{
    m_active=true;
}


/**
 * @name    operator()
 * @brief   copy the characters from input buffer to the message buffer
 *
 * @param [in]  buffer              pointer to the buffer
 * @param [in]  bytes_transferred   nr. byte transferred 
 *
 * @retval none.
 */	
void ResponseHandler::operator()(const char* buffer,const size_t bytes_transferred){
    for (unsigned int i=0; i < bytes_transferred ;++i){
        read_msgs_.push_back(buffer[i]);
    }
}


/**
 * @name    _run
 * @brief   while the m_actice is true, the _run executing cyclically, 
 *          read a character from the message buffer and sends it to the processer.
 *
 * @retval none.
 *
 * Example Usage:
 * @code
 *    boost::thread t(boost::bind(&ResponseHandler::_run, &responseHandler)
 * @endcode
 */	
void ResponseHandler::_run(){
    while(m_active){
        if(!read_msgs_.empty()){
            char l_c=read_msgs_.front();
            read_msgs_.pop_front();
            processChr(l_c);
        }
    }
}


/**
 * @name    createCallbackFncPtr 
 * @brief   static function
 *          Create a callback function object, through which can be called a function for handling certain message responses.
 *
 * @retval new object
 *
 * Example Usage:
 * @code
 *     ResponseHandler::CallbackFncPtrType l_callbackFncObj=ResponseHandler::createCallbackFncPtr(&functionName);
 *     (*l_callbackFncObj)(response);
 * @endcode
 */
ResponseHandler::CallbackFncPtrType ResponseHandler::createCallbackFncPtr(void (*f)(std::string)){
    return  new CallbackFncType(f);
}


/**
 * @name    attach
 * @brief   Attach the callback function  to the response key word. This callback function will be called automatically, when will be received the key word 
 *          feedback from the Embedded. More functions can be attach  to the one key word. 
 *
 * @retval new object
 *
 * Example Usage:
 * @code
 *     l_responseHandler.attach(message::KEY,l_callbackFncObj)
 * @endcode
 */
void ResponseHandler::attach(std::string f_action,CallbackFncPtrType waiter){
    if(m_keyCallbackFncMap.count(f_action)>0){
        CallbackFncContainer* l_container=&(m_keyCallbackFncMap[f_action]);
        l_container->push_back(waiter);
    }else{
        CallbackFncContainer l_container;
        l_container.push_back(waiter);
        m_keyCallbackFncMap[f_action]=l_container;
    }
}


/**
 * @name    detach
 * @brief   After applying detach  on a certain function and message. The callback function will not be called anymore after applying this.
 *
 * @retval new object
 *
 * Example Usage:
 * @code
 *      l_responseHandler.attach(message::KEY,l_callbackFncObj)
 * @endcode
 */
void ResponseHandler::detach(std::string f_action,CallbackFncPtrType waiter){
    if(m_keyCallbackFncMap.count(f_action)>0){
        CallbackFncContainer *l_container=(&m_keyCallbackFncMap[f_action]);
        CallbackFncContainer::iterator it=std::find(l_container->begin(),l_container->end(),waiter);
        if(it!=l_container->end()){  
            l_container->erase(it);
        } 
        else{
            std::cout<<"Not found!"<<std::endl;   
        }
        
    }else{
        std::cout<<"Container is empty!"<<std::endl;   
    }
}


/**
 * @name    processChr
 * @brief   Each received character is sent to this function. If the char is '@', it signales the begining of the response.
 *          If the character is new line ('/r'), it signales the end of the response. 
 *          If there is any other character, it appends it to the valid response attribute. .
 *
 * @retval new object
 *
 * Example Usage:
 * @code
 *      l_responseHandler.attach(message::KEY,l_callbackFncObj)
 * @endcode
 */
void ResponseHandler::processChr(const char f_received_chr){
    if (f_received_chr=='@'){
        m_isResponse=true;
    }
    else if(f_received_chr=='\r'){
        if (!m_valid_response.empty()){
            checkResponse();
            m_valid_response.clear();
        }
        m_isResponse=false;
    }
    if(m_isResponse){
        m_valid_response.push_back(f_received_chr);
    }                
}

 /*  
 * Response example: "@1:RESPONSECONTANT;;\r\n"
 */
void ResponseHandler::checkResponse(){
    std::string l_responseFull(m_valid_response.begin(),m_valid_response.end());
    std::string l_keyStr=l_responseFull.substr(1,1);
    std::string l_reponseStr=l_responseFull.substr(3,l_responseFull.length()-5);
    if(std::stoi(l_keyStr)>0){
        CallbackFncContainer l_cointaner=m_keyCallbackFncMap[l_keyStr];
        for(CallbackFncContainer::iterator it=l_cointaner.begin();it!=l_cointaner.end();++it){
            (**it)(l_reponseStr);
        }
    }
    else{
        std::cout<<"";
    }
}

