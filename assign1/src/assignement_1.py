import time

def on_mult(m_ar, b_ar):
    pha = [0.0] * (m_ar * m_ar)
    phb = [0.0] * (b_ar * b_ar)
    phc = [0.0] * (m_ar * m_ar)

    # Fill A with 1.0
    for i in range(m_ar):
        for j in range(b_ar):
            pha[i * m_ar + j] = 1.0

    # Fill B with (i + 1)
    for i in range(b_ar):
        for j in range(b_ar):
            phb[i * m_ar + j] = float(i + 1) 

    start = time.perf_counter()

    for i in range(m_ar):
        for j in range(m_ar):
            temp = 0.0
            for k in range(m_ar):
                temp += pha[i * m_ar + k] * phb[k * b_ar + j]
            phc[i * m_ar + j] = temp

    end = time.perf_counter()

    print("Time:", end - start, "seconds")
    print("Result matrix:")

    # Print first row, first min(10, b_ar) elements (like your JS)
    for j in range(min(10, b_ar)):
        print(phc[j], end=" ")
    print()

def onMultLine(m_ar, m_br):
    pha = [0.0] * (m_ar * m_ar)
    phb = [0.0] * (m_br * m_br)
    phc = [0.0] * (m_ar * m_ar)

    # Fill A with 1.0
    for i in range(m_ar):
        for j in range(m_br):
            pha[i * m_ar + j] = 1.0

    # Fill B with (i + 1)
    for i in range(m_br):
        for j in range(m_br):
            phb[i * m_ar + j] = float(i + 1) 
    
    start = time.perf_counter()
    
    for i in range (m_br):
        for k in range (m_ar):
            temp = pha[i*m_ar + k]
            for j in range (m_br):
                phc[i*m_ar + j] += temp * phb[k*m_ar+j]

    end = time.perf_counter()

    print("Time:", end - start, "seconds")
    print("Result matrix:")

    for j in range(min(10, m_br)):
        print(phc[j], end=" ")
    print()

def onMultBlock(m_ar, m_br, bk_size):
    pha = [0.0] * (m_ar * m_ar)
    phb = [0.0] * (m_br * m_br)
    phc = [0.0] * (m_ar * m_ar)

    # Fill A with 1.0
    for i in range(m_ar):
        for j in range(m_br):
            pha[i * m_ar + j] = 1.0

    # Fill B with (i + 1)
    for i in range(m_br):
        for j in range(m_br):
            phb[i * m_ar + j] = float(i + 1) 

    start = time.perf_counter()


    for ii in range(0, m_ar, bk_size):
        iMax = min(ii + bk_size, m_ar)
        for kk in range(0, m_ar, bk_size):
            kMax = min(kk + bk_size, m_ar)
            for jj in range(0, m_ar, bk_size):
                jMax = min(jj + bk_size, m_ar)
                for i in range(ii, iMax):
                    iN = i * m_ar
                    for k in range(kk, kMax):
                        a = pha[iN + k]
                        kN = k * m_ar
                        for j in range(jj, jMax):
                            phc[iN + j] += a * phb[kN + j]

    end = time.perf_counter()

    print("Time:", end - start, "seconds")
    print("Result matrix:")

    for j in range(min(10, m_br)):
        print(phc[j], end=" ")
    print()

onMultBlock(1024, 1024, 64)