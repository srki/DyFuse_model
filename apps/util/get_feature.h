#define NUM_FEATURE 4

#define ABORT(err_msg) \
 { char msg[256];\
   sprintf(msg,"%s at line %d in file %s\n",err_msg,__LINE__, __FILE__);\
   printf("%s", msg);  \
   exit(-1);}
