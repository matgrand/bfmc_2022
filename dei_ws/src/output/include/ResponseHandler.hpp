#ifndef MESSAGE_HANDLER_HPP
#define MESSAGE_HANDLER_HPP

#include <iostream>
#include <deque>
#include <map>
#include <vector>

#include <boost/function.hpp>
#include <Message.hpp>

class BaseResponseHandler{
    public:
        virtual void  operator()(const char*,const size_t)=0;
        virtual void _run()=0;
        virtual void deactive();
    protected:
        bool             m_active;//remain true, while the _run executing
};

class ResponseHandler:public BaseResponseHandler{
    public:
        typedef boost::function<void(std::string)>      CallbackFncType;
        typedef boost::function<void(std::string)>*     CallbackFncPtrType;
        typedef std::vector<CallbackFncPtrType>         CallbackFncContainer;
        
        template<class T>
        static CallbackFncPtrType createCallbackFncPtr(void (T::*f)(std::string),T* obj){
            return  new CallbackFncType( std::bind1st(std::mem_fun(f),obj));
        }
        static CallbackFncPtrType createCallbackFncPtr(void (*f)(std::string));
        
        ResponseHandler();
        
        void operator()(const char*,const size_t);
        void _run();

        void attach(std::string,CallbackFncPtrType);
        void detach(std::string,CallbackFncPtrType);
        
    private:
        void processChr(const char);
        void checkResponse();

        
        std::deque<char> read_msgs_;  //buffered read data
        bool             m_isResponse;//is true, when receiving a response valid from the device 
        std::deque<char> m_valid_response;//buffer the valid response
        std::map<std::string,CallbackFncContainer>  m_keyCallbackFncMap;

        
};

#endif
