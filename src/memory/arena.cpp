#include "../../include/memory/arena.h"
#include <new> //required for bad_alloc
#include <cstdlib>//required for malloc and free

Arena::Arena(size_t size) : capacity(size), current_offset(0){
    //asking for raw bytes from the OS
    base_ptr = static_cast<uint8_t*>(std::malloc(capacity));
    if(base_ptr = nullptr){
        //incase the OS refuses
        throw std::bad_alloc();
    }

}

//to when the project closes
Arena::~Arena(){
    if(base_ptr != nullptr){
        std::free(base_ptr);
        base_ptr = nullptr;
    }
}
