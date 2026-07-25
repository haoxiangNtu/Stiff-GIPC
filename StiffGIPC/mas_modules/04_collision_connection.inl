__global__ void _buildCollisionConnection_new(unsigned int* _pConnect,
                                              const int*    _pCoarseSpaceTable,
                                              const const int4* _collisionPair,
                                              const int* _real_map_partId,
                                              int        level,
                                              int        node_offset,
                                              int        vertNum,
                                              int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4 MMCVIDI              = _collisionPair[idx];
    int* collitionPairStartId = &(MMCVIDI.x);
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w < 0)
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
        }

        for(int i = 0; i < 4; i++)
            collitionPairStartId[i] -= node_offset;

        int cpVertNum = 4;
        int cpVid[4];
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] =
                        _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                else
                    cpVid[i] = -1;
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                else
                    cpVid[i] = -1;
        }

        unsigned int connMsk[4] = {0};

        for(int i = 0; i < 4; i++)
        {
            for(int j = i + 1; j < 4; j++)
            {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if(myId == otId || myId < 0 || otId < 0)
                {
                    continue;
                }
                if(myId / BANKSIZE == otId / BANKSIZE)
                {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(cpVid[i] >= 0)
                {
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
                }
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                {
                    atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                }
        }
    }
    else
    {
        int v0I   = -MMCVIDI.x - 1;
        MMCVIDI.x = v0I;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 2;
                int cpVid[2];

                for(int i = 0; i < 2; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[2] = {0};

                for(int i = 0; i < 2; i++)
                {
                    for(int j = i + 1; j < 2; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;
                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 3;
                int cpVid[3];
                for(int i = 0; i < 3; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[3] = {0};

                for(int i = 0; i < 3; i++)
                {
                    for(int j = i + 1; j < 3; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else
        {
            int cpVertNum = 4;
            int cpVid[4];
            for(int i = 0; i < 4; i++)
                collitionPairStartId[i] -= node_offset;
            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                    else
                        cpVid[i] = -1;
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                    else
                        cpVid[i] = -1;
            }

            unsigned int connMsk[4] = {0};

            for(int i = 0; i < 4; i++)
            {
                for(int j = i + 1; j < 4; j++)
                {
                    unsigned int myId = cpVid[i];
                    unsigned int otId = cpVid[j];

                    if(myId == otId || myId < 0 || otId < 0)
                    {
                        continue;
                    }
                    if(myId / BANKSIZE == otId / BANKSIZE)
                    {
                        connMsk[i] |= (1U << (otId % BANKSIZE));
                        connMsk[j] |= (1U << (myId % BANKSIZE));
                    }
                }
            }

            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(cpVid[i] >= 0)
                    {
                        atomicOr(_pConnect + cpVid[i], connMsk[i]);
                    }
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                    {
                        atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                    }
            }
        }
    }
}


// [per-env MAS] Env-segment an already-exclusive-scanned per-warp prefix so each env's coarse
// clusters occupy a BANKSIZE-aligned block → envs never share a bank at any level → the
// block-diagonal Schwarz smoother stays intra-env → strict cross-env / batch bit-identity.
// Homogeneous multi-env: env e owns warps [e*wpe,(e+1)*wpe). ALL device-side (integer, exact →
// zero determinism impact; no host round-trip, no sync — the earlier host cudaMemcpy version was
// only an implementation shortcut, not a determinism requirement).
//
// _mas_env_base: n_env is small → single thread. Computes each env's aligned base offset + its
// pre-mutation scan start (envStart, so _apply has no read-after-write hazard) + the padded total.
