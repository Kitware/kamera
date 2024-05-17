#!/usr/bin/env python

import os, sys, time, mmap
import numpy as np
import cv2


def new_img(width=6576, height=4384, chan=3):
    return np.random.randint(0, 255, size=(width, height, chan), dtype=np.uint8)


def time_n_image(path='/dev/shm/img', n=12):
    arys = [new_img() for _ in range(n)]
    elapsed = 0
    start = time.time()
    for i, a in enumerate(arys):
        fn = path + '{}.bmp'.format(i)
        cv2.imwrite(fn, a)
        with open(fn, 'rb') as fp:
            print(os.fstat(fp))
        b = cv2.imread(fn)
        if a.shape != b.shape:
            print('Failed to extract same image')

    end = time.time()
    elapsed += end - start
    mbps = elapsed * n * 8 * arys[0].size / 1e6
    print('{} count {} bytes {} s {} each {} Mbps'.format(n, a.size, elapsed, elapsed / n, mbps))


def time_n_fp(path='/dev/shm/img', n=12):
    arys = [new_img() for _ in range(n)]
    elapsed = 0
    start = time.time()
    for i, a in enumerate(arys):
        fn = path + '{}.npy'.format(i)
        with open(fn, 'wb') as fp:
            np.save(fp, a)
            os.fsync(fp)

    end = time.time()
    elapsed += end - start
    mbps = elapsed * n * 8 * arys[0].size / 1e6
    print('{} count {} bytes {} s {} each {} Mbps'.format(n, arys[0].size, elapsed, elapsed / n, mbps))


def time_n_mm(path='/dev/shm/img', n=12, period=0.125):
    arys = [new_img() for _ in range(n)]
    elapsed = 0
    run_elapsed = 0
    write_elapsed = 0
    for i, img in enumerate(arys):
        start = time.time()
        fn = path + '{}.bin'.format(i)
        fpo = np.memmap(fn, dtype='uint8', mode='write', shape=img.shape)
        fpo[:] = img[:]
        del fpo
        written = time.time()
        run_elapsed = written - start
        write_elapsed += run_elapsed
        dt = period - run_elapsed
        if dt > 0:
            time.sleep(period - run_elapsed)
        with open(fn, 'rb') as fp:
            fd = fp.fileno()
            os.fsync(fd)
            st = os.fstat(fd)
            if st.st_size != img.size:
                print('Size mismatch: {} vs {}'.format(img.size, st.st_size))

        end = time.time()
        elapsed += end - start
    mbps = (n * 8 * arys[0].size / 1e6) / write_elapsed
    effective_mbps = (n * 8 * arys[0].size / 1e6) / elapsed
    print(arys[0].size)
    print('write  {} count {: >9} bytes {:.3f} s {:.3f} each {:.1f} Mbps'.format(n, arys[0].size, elapsed, write_elapsed / n, mbps))
    print('effctv {} count {: >9} bytes {:.3f} s {:.3f} each {:.1f} Mbps'.format(n, arys[0].size*n, elapsed, elapsed / n, effective_mbps))


if __name__ == '__main__':
    assert len(sys.argv) > 1, "specify a shm file path location"
    fn = sys.argv[1]
    try:
        os.makedirs(os.path.dirname(fn))
    except Exception as exc:
        print('{}: {}'.format(exc.__class__.__name__, exc))
    print(fn)
    time_n_mm(fn, 1)
    time_n_mm(fn, 2)
    time_n_mm(fn, 6)
    time_n_mm(fn, 8)
    time_n_mm(fn, 12)
    time_n_mm(fn, 16)
    time_n_mm(fn, 24)

