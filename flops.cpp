#include "flops.h"

FlopsCounter* getCounter() {
    static FlopsCounter ctr;

    return &ctr;    
}