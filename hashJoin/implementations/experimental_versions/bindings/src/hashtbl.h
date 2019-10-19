/*
 *  File: hashtbl.h
 */
#ifndef HASHTBL_H
#define HASHTBL_H

struct HashtblConfig {
    unsigned table_size;
    unsigned cuda_blocksize;
    unsigned cuda_numblocks;
};

const unsigned EmptyEntry = 0xffffffffu;
typedef unsigned long long Entry;  // key 32bit | value 32bit

class Hashtbl {
   public:
    // Allocate Memory for the Hashtbl with the size given in HashtblConfig
    Hashtbl(HashtblConfig config);

    // Allow to insert multiple items at a time
    //virtual const bool Insert(unsigned input_size, unsigned *keys,
     //                         unsigned *vals);

    // Allow to retrive multiple items at a time
	bool Build(const unsigned input_size, unsigned *keys,
                        unsigned *vals);

	bool Probe(const unsigned input_size, unsigned *keys,
                        unsigned *vals, unsigned *output_size,
                        unsigned *output);

    ~Hashtbl();

    void DumpHashtbl();
    
    inline unsigned get_table_size() const { return table_size; }
    inline unsigned get_number_of_elements() const {
        return number_of_elements;
    }

    unsigned *content;

   protected:
    unsigned table_size;
    unsigned number_of_elements;
};

__global__ 
void build(const unsigned input_size, unsigned *content,
                      const unsigned table_size, const unsigned *keys,
                      const unsigned *vals);


__global__ 
void build_linprobe(const unsigned input_size, unsigned *content,
                      const unsigned table_size, const unsigned *keys,
                      const unsigned *vals);

__global__ 
void build_linprobe(const int input_size, int *content,
                      const uint64_t table_size, const int *keys,
                      const int *vals);

__global__ 
void build_linprobe_lsb(const int input_size, int *content,
                      const uint64_t table_size, const int *keys,
                      const int *vals);

__global__
void probe_linprobe_lsb(const int input_size,
                      const int *keys, const int *vals,
                      int *content, const int table_size,
                      int *output, int *current);

__global__ 
void build_linprobe_shared(const int input_size,
                      const uint64_t table_size, const int *keys,
                      const int *vals);


__global__
 void probe(const unsigned input_size,
                      const unsigned *keys, const unsigned *vals,
                      unsigned *content, const unsigned table_size,
                      unsigned *output, unsigned *current);
__global__
void build_and_probe_sm(const unsigned table_size,
                     const unsigned input_size_build,
                     const unsigned *keys_build, const unsigned *vals_build,
                     const unsigned input_size_probe,
                     const unsigned *keys_probe, const unsigned *vals_probe,
                     unsigned *output,
					 unsigned *current,
					 unsigned *content_);
#endif /* HASHTBL_H*/
