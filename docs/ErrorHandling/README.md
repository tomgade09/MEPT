# ErrorHandling


### What is?
This folder contains a number of error handling macros designed to catch exceptions, and ensure certain conditions are met, as well as two custom exception classes.


### Exception Classes
```
class SimException : public std::exception
{
   SimException(std::string error, std::string fileName, int lineNum, std::vector<double> numArgs = {}, std::vector<std::string> strArgs = {})
   /*
      ...
   */
};
```
Exception class that allows for specifying function arguments as part of the data captured by a throw.  Class stores an error string (explaining what went wrong), the file name (typically with the `__FILE__` macro), a line number (typically with the `__LINE__` macro), number arguments, and string arguments.  Has `.what()` function (like `std::exception`), as well as `.where()` ([file name]:[line number]) and `.args()` (arguments that you pass in).  Best used with the `SIM_API_EXCEP_CHECK` macro to catch.


```
class SimFatalException : public SimException
{
   SimFatalException(std::string error, std::string fileName, int lineNum, std::vector<double> numArgs = {}, std::vector<std::string> strArgs = {})
   /*
      ...
   */
};
```
Exactly the same as above, except the `SIM_API_EXCEP_CHECK` macro also includes `exit(1)`, exiting the application (somewhat) gracefully, when catching this one.


### Macros
```
SIM_API_EXCEP_CHECK( /*YOUR CODE HERE*/ );
```
Wrap code in this to catch `SimException` and `SimFatalException` as well as specifically `std::invalid_argument`, `std::out_of_range`, `std::logic_error`, `std::runtime_error`, and generally `std::exception` and print the `.what()` along with some other data to `cerr`.  Used in conjunction with capturing cerr can save this to a log file.


```
CUDA_API_ERRCHK( /*CUDA API FUNCTION CALL HERE*/ );
```
Wrapper to check if CUDA API function completes without error.  If there's an error, prints to `cerr`.  Used in conjunction with capturing cerr can save this to a log file.


```
/*YOUR CUDA KERNEL CALL HERE*/
CUDA_KERNEL_ERRCHK();
```
Macro (placed **_after_** a global kernel call) that checks for an error with the previous kernel's execution.  Use this if you don't want the overhead of a `cudaDeviceSynchronize` after the kernel call.  If there's an error, prints to `cerr`.  Used in conjunction with capturing cerr can save this to a log file.


```
/*YOUR CUDA KERNEL CALL HERE*/
CUDA_KERNEL_ERRCHK_WSYNC();
```
Macro that functions the same as above with a `cudaDeviceSynchronize()` before checking for an error.  Use this by default as long as a sync operation has negligible impact on performance.


```
/*YOUR CUDA KERNEL CALL HERE*/
CUDA_KERNEL_ERRCHK_WSYNC_WABORT();
```
Macro that adds `exit(1)` if an error is encountered to the above macro.


```
/*YOUR CUDA KERNEL CALL HERE*/
CUDA_KERNEL_ERRCHK_WABORT()
```
Macro that's same as above, except *without* sync, but with abort.


```
ZEROTH_THREAD_ONLY("NAME OF FUNCTION HERE", /*YOUR CUDA KERNEL CODE HERE*/);
```
Wrap kernel (i.e. **_on-device only_**) code in this to prevent execution on multiple threads.  Ensures `threadIdx.x` and `blockIdx.x` are zero, otherwise an error message is printed and the function is returned.


[Up a level](./../README.md)