## Run:

I'm using camera capture for testing - it is more interesting to test than a video file.
I'll supply a screencast of running the program, so you will not need to run it on your machine.

- Install poetry if does not installed: `pip install poetry`
- `poetry install`
- `poetry run python .`

## Results:

### Performance:

- KCF - fastest
- CSRT - medium
- MIL - very slow

### Target loss:

- KCF - lost target often
- MIL - lost target during fast movement
- CSRT - most stable, but can "jump" on the other object during fast movement

### Conclusion:

CSRT is the best compromise between speed and stability. I'm working with ConstantRobotic's derivative of CSRT, which is faster than the original OpenCV CSRT implementation. Still, the problems are the same: it can "jump" to the other object during rapid movement.

Using a grayscale image for tracking is not a good idea, as KCF does not support it. Additionally, MIL and CSRT appear less stable on grayscale images, although their performance is almost the same.

### What's next:

I want to use multiple KCF trackers for optical flow image stabilization in my work project because it is fast. But I need to find a way to make it more stable.
