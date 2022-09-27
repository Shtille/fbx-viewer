#!/bin/bash
# Script for resource syncronization between 'scythe/data' and 'data'.
# 'data' fully includes 'scythe/data' and extends it.
# Should be called from this directory.
rsync -avz deps/scythe/data/ ./data