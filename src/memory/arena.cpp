#include "../../include/memory/arena.h"
#include <new> //required for bad_alloc
#include <windows.h>//talks directly to the kernal

Arena::Arena(size_t size) : capacity(size), current_offset(0){
    //asking for raw bytes from the OS
    base_ptr = static_cast<uint8_t*>(VirtualAlloc(NULL, capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if(base_ptr == nullptr){
        //incase the OS refuses
        throw std::bad_alloc();
    }

}

//to when the project closes
Arena::~Arena(){
    if(base_ptr != nullptr){
        VirtualFree(base_ptr, 0, MEM_RELEASE);
        base_ptr = nullptr;
    }
}
