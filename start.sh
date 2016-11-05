# Starts generating and talking!
nvidia-docker run --rm -it albarji/neurocervantes "En un lugar de la Mancha" | tee >(espeak -ves+m3 -p1 -s30)
