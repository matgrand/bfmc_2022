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

#include "serialPortHandler.hpp"

/*
* This class handles the sending of the messages to the Embedded via serial and it forwards the received messages to the ResponseHandler
*/

using namespace std;

/**
 * @name    Class constructor
 * @brief   
 *
 * Open a serial port and perform flush for both receive and transmit buffers
 *
 * @param [in] io_service  Object that provides core I/O functionality.
 * @param [in] baud        Baud rate.
 * @param [in] device      Device name (listed in /dev folder on RPi)
 * @param [in] responseHandler   Object that provides response message processing 
 *
 * @retval none
 *
 * Example Usage:
 * @code
 *    serialPortHandler c(io_service, boost::lexical_cast<unsigned int>(argv[1]), argv[2]);
 * @endcode
 */
serialPortHandler::serialPortHandler(boost::asio::io_service& io_service, unsigned int baud, const string& device,BaseResponseHandler& responseHandler)
		: m_responseHandler(responseHandler),
		  active_(true),
		  io_service_(io_service),
		  serialPort(io_service, device)
		 

{
	if (!serialPort.is_open())
	{
		cerr << "Failed to open serial port\n";
		return;
	}
	else
	{
		cout << "Port opened!"<< std::endl;
	}

	boost::asio::serial_port_base::baud_rate baud_option(baud);
	serialPort.set_option(baud_option); // set the baud rate after the port has been opened

	boost::system::error_code error;
	flush_serial_port(serialPort,flush_both,error);
	std::cout << "flush: " << error.message() << std::endl;

	read_start();
}


/**
 * @name    write
 * @brief   Send data over UART.
 *
 *  pass the write data to the do_write function via the io service in the other thread
 *
 * @param [in] smg  Message to be sent.
 *
 * @retval none.
 *
 * Example Usage:
 * @code
 *    c.write(msg);
 * @endcode
 */
void serialPortHandler::write(std::string f_msg) 
{
	io_service_.post(boost::bind(&serialPortHandler::do_writeString, this, f_msg));
}	



// call the do_close function via the io service in the other thread
void serialPortHandler::close() 
{
	io_service_.post(boost::bind(&serialPortHandler::do_close, this, boost::system::error_code()));
}

// return true if the socket is still active
bool serialPortHandler::active() 
{
	return active_;
}

// Start an asynchronous read and call read_complete when it completes or fails
void serialPortHandler::read_start(void)
{ 
	serialPort.async_read_some(boost::asio::buffer(read_msg_, max_read_length),
		boost::bind(&serialPortHandler::read_complete,
			this,
			boost::asio::placeholders::error,
			boost::asio::placeholders::bytes_transferred));
}

// the asynchronous read operation has now completed or failed and returned an error
void serialPortHandler::read_complete(const boost::system::error_code& error, size_t bytes_transferred)
{ 
	if (!error)
	{ // read completed, so process the data
		
		m_responseHandler(read_msg_,bytes_transferred);
		// cout.write(read_msg_, bytes_transferred); // echo to standard output
		read_start(); // start waiting for another asynchronous read again
	}
	else
		do_close(error);
}

// callback to handle write call from outside this class
void serialPortHandler::do_writeString(std::string f_msg){
	
	boost::lock_guard<boost::mutex>* guard =new boost::lock_guard<boost::mutex>(m_writeMtx);
	bool write_in_progress = write_msgs_.empty();
	for ( std::string::iterator it=f_msg.begin(); it!=f_msg.end(); ++it)
	{
		write_msgs_.push_back(*it);
	}
	delete guard;
	if(write_in_progress){
		write_start();
	}
}

// Start an asynchronous write and call write_complete when it completes or fails
void serialPortHandler::write_start(void)
{ 
	boost::asio::async_write(serialPort,
		boost::asio::buffer(&write_msgs_.front(), 1),
		boost::bind(&serialPortHandler::write_complete,
			this,
			boost::asio::placeholders::error));
}

// the asynchronous read operation has now completed or failed and returned an error
void serialPortHandler::write_complete(const boost::system::error_code& error)
{
	if (!error)
	{ // write completed, so send next write data
		write_msgs_.pop_front(); // remove the completed data
		if (!write_msgs_.empty()) // if there is anthing left to be written
			write_start(); // then start sending the next item in the buffer
	}
	else
		do_close(error);
}

// something has gone wrong, so close the socket & make this object inactive
void serialPortHandler::do_close(const boost::system::error_code& error)
{ 
	if (error == boost::asio::error::operation_aborted) // if this call is the result of a timer cancel()
		return; // ignore it because the connection cancelled the timer
	if (error)
		cerr << "Error: " << error.message() << endl; // show the error message
	serialPort.close();
	active_ = false;
}

/// @brief Flush a serial port's buffers.
///
/// @param serial_port Port to flush.
/// @param what Determines the buffers to flush.
/// @param error Set to indicate what error occurred, if any.
void serialPortHandler::flush_serial_port(
	boost::asio::serial_port& serial_port, 
	flush_type what, 
	boost::system::error_code& error)
{
	if (0 == ::tcflush(serial_port.lowest_layer().native_handle(), what))
  	{
    	error = boost::system::error_code();
  	}
  	else
  	{
    	error = boost::system::error_code(
			errno,
        	boost::asio::error::get_system_category());
  	}
}
