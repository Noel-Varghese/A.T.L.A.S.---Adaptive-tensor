#pragma once //tells the compiler to only include this file once to prevent duplicates
#include <cstdint>//gives the exact width integers like uint_8 which is exactily one byte
#include <cstddef> // Gives us size_t, which is standard for c++ memory size
#include <stdexcept>//allows us to throw exception and bad_alloc errors

class Arena{
    private:
        //Starting address of a memory block
        uint8_t* base_ptr;//helps move pointer byte by byte
        size_t capacity;//maximum size 
        size_t current_offset;//tracks the number of bytes used so far
    public:
        Arena(size_t size);//used to ask OS for RAM

        ~Arena();//deconstructor
    //returns memory address for the tensor
    //request_size = how many bytes a tensor needs
    //alignment = hardware ruke(default 32 bytes for AVX2)
    void* allocate(size_t request_size, size_t alignment = 32){
        //the math used:
        /* alignment_storage = (current_address + (alignment-1))&~(alinment-1) */
        size_t alg_offset = (current_offset + (alignment -1 )&~(alignment -1));
        if(alg_offset + request_size > capacity){
            throw std::bad_alloc();
        }
        void* result_ptr = base_ptr +alg_offset;//calculates exact memory address
        current_offset = alg_offset + request_size;
        return result_ptr;//gives back the memory address
    }
    void reset(){//helps overwrite old data
        current_offset = 0;
    }
};