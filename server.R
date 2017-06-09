# Code from 
# https://www.r-bloggers.com/image-classification-in-r-using-trained-tensorflow-models/
#

library(shiny)
library(tensorflow)
library(magrittr)
library(jpeg)
library(grid)
library(ggplot2)

plot_jpeg = function(path, add=FALSE)
{
  require('jpeg')
  jpg = readJPEG(path, native=T) # read the file
  res = dim(jpg)[1:2] # get the resolution
  if (!add) # initialize an empty plot area if add==FALSE
    plot(1,1,xlim=c(1,res[1]),ylim=c(1,res[2]),asp=1,type='n',xaxs='i',yaxs='i',xaxt='n',yaxt='n',xlab='',ylab='',bty='n')
  rasterImage(jpg,1,1,res[1],res[2])
}

globalFunction <- function(imgs) {
  
  print("loading... define_network_shape function")
  
  define_network_shape <- function(){
    # Define the shape of the network.
    fc8 = slim$conv2d(imgs_scaled, 64, shape(3,3), scope='vgg_16/conv1/conv1_1') %>%
      slim$conv2d(64, shape(3,3), scope='vgg_16/conv1/conv1_2') %>% 
      slim$max_pool2d(shape(2,2), scope='vgg_16/pool1') %>%
      
      slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_1')  %>%
      slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_2')  %>%
      slim$max_pool2d( shape(2, 2), scope='vgg_16/pool2')  %>%
      
      slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_1')  %>%
      slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_2')  %>%
      slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_3')  %>%
      slim$max_pool2d(shape(2, 2), scope='vgg_16/pool3')  %>%
      
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_1')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_2')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_3')  %>%
      slim$max_pool2d(shape(2, 2), scope='vgg_16/pool4')  %>%
      
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_1')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_2')  %>%
      slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_3')  %>%
      slim$max_pool2d(shape(2, 2), scope='vgg_16/pool5')  %>%
      
      slim$conv2d(4096, shape(7, 7), padding='VALID', scope='vgg_16/fc6')  %>%
      slim$conv2d(4096, shape(1, 1), scope='vgg_16/fc7') %>% 
      
      # Setting the activation_fn=NULL does not work, so we get a ReLU
      slim$conv2d(1000, shape(1, 1), scope='vgg_16/fc8')  %>%
      tf$squeeze(shape(1, 2), name='vgg_16/fc8/squeezed')
  }
  
  print("Import slim library and initialize")
  slim = tf$contrib$slim
  tf$reset_default_graph()
  images = tf$placeholder(tf$float32, shape(NULL,NULL,NULL,3))
  imgs_scaled = tf$image$resize_images(images, shape(224,224))
  
  fc8 <- define_network_shape()
  print("Start session and restore the model from a file")
  restorer = tf$train$Saver()
  sess = tf$Session()
  #  sess <- tf$InteractiveSession()
  restorer$restore(sess, 'vgg_16.ckpt')
  
  
  return(sess$run(fc8, dict(images=imgs)))
}  

shinyServer(function(input, output) {
  
  # The function for plotting the image with classifications
  output$classifImage <- renderPlot({
    inData <- input$img
    if (is.null(inData))
      return(NULL)
    img1 <- readJPEG(inData$datapath)
    d <- dim(img1)
    imgs <- array(255*img1, dim=c(1, d[1], d[2], d[3]))
    fc8_vals <- globalFunction(imgs)
    probs <- exp(fc8_vals)/sum(exp(fc8_vals))
    idx <- head(order(probs,decreasing=T), n=5)
    names = read.delim("imagenet_classes.txt", header=F)
    
    g = rasterGrob(img1, interpolate=TRUE) 
    text = c()
    #for (id in idx) {
    #  text = paste0(text, names[id,][[1]], " ", round(probs[id],5), "\n") 
    #}
    prob = c()
    for (id in idx) {
      text = c(text, as.character(names[id,][[1]]))
      prob = c(prob, round(probs[id],2)) 
    }
    
    output$classifs <- renderTable({
      data.frame(label=text,prob=prob)
    })
    
    ggplot(data.frame(d=1:3)) + annotation_custom(g) # + 
    #  annotate('text',x=0.05,y=0.05,label=text, size=7, hjust = 0, vjust=0, color='blue') + xlim(0,1) + ylim(0,1) 
  })
})
