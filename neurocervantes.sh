#!/bin/bash
#
# Starts generating Neurocervantes text!
# And speaks it out loud!!
#
# Arguments
#	$1: seed to use for the generation
#

MODELFILE="elquijote"
TEMPERATURE=0.5
MAXCHARS=10000
THEANO_FLAGS='device=gpu' python neurowriter-generate.py $MODELFILE $TEMPERATURE "$1" $MAXCHARS
