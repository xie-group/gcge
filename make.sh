#!/bin/sh
mpicc slepcgcge.c  -o slepcgcge -lgcge -I./src -I./app
