#!/bin/bash

./cuda29 -n 671 > /dev/null
#md5sum SeedA-bufferAB.bin
#md5sum SeedA-indexesE1.bin
#md5sum SeedB-bufferA.bin
#md5sum SeedB-indexesE0.bin
#md5sum Round0-indexesE1.bin
#md5sum Round0-indexesE2.bin
#md5sum Round1-indexesE0.bin
md5sum Round175-indexesE0.bin
