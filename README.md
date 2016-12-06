# shiny-imageclassif-tfslim

This is a Shiny version of the example VGG16 classifier that uses TF-Slim in R described in the blog post at https://www.r-bloggers.com/image-classification-in-r-using-trained-tensorflow-models/.

For it to work after cloning, you need to download and extract the VGG16 model checkpoint file:

http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

...into the same directory as the other files.

You should be able to run this as a normal Shiny application, e g by shiny::runApp() or the Run App button in RStudio, if you are standing in the right directory.