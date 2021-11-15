# Running the code on our servers eduserv.flov.gu.se and mltgpu.flov.gu.se

## Image classification and OpenCV

(adapted from email from Robert Adessam)

To run `*.show()` which shows an image in a graphical window you need to install an `xserver` on your local machine and login with `ssh -Y` (at least twice as the first time will create the needed Xauthority file).

For Linux you do not need anything.

For Mac I recommend Xquarz: https://www.xquartz.org/

For Windows I don't really know but maybe VcXsrv: https://sourceforge.net/projects/vcxsrv/
