#ifndef MASTER_H
#define MASTER_H

#include <stdio.h>

#include "ringbuffer.h"
#include "data.h"

enum processing_mode_e{cpu_mode, gpu_mode};


struct master_ctx_t {

	/** This queue handles the imput data of the R stream **/
	ringbuffer_t *data_R_queue;

	/** This queue handles the imput data of the S stream **/
	ringbuffer_t *data_S_queue;

	/** Result tuples are stored in this queue **/
	ringbuffer_t *result_queue;

	/** data rate  of the R stream **/
	unsigned rate_R;

	/** data rate of the S stream **/
	unsigned rate_S;

	/** window size for R (in sec) **/
	unsigned window_size_R;

	/** window size for S (in sec) **/
	unsigned window_size_S;
	
	/** number of tuples to pregenerate for R **/
	unsigned num_tuples_R;

	/** number of tuples to pregenerate for S **/
	unsigned num_tuples_S;

	/** value range for the join attribute of type @c int (x and a) **/
	unsigned int_value_range;

	/** value range for the join attribute of type @c float (y and b) **/
	unsigned float_value_range;

	/** File to write the result tuples to  **/
	FILE *resultfile;

	/** File to write the output **/
	FILE *outfile;

	FILE *logfile;

	/**
	 * Input data stream R
	 */
	struct {
		struct timespec *t;
		x_t             *x;
		y_t             *y;
		z_t             *z;
	} R;

	/**
	 * Input data stream S
	 */
	struct {
		struct timespec *t;
		a_t             *a;
		b_t             *b;
		c_t             *c;
		d_t             *d;
	} S;

    /**
     * whether the join is processed on the cpu or gpu
     */
    processing_mode_e processing_mode;

	//struct {
    //       unsigned int   pos;   /**< number of tuples already in the message */
    //       result_msg_t   msg;   /**< the partial message itself */
    //} partial_result_msg;
};
typedef struct master_ctx_t master_ctx_t;

#endif /* MASTER_H */
