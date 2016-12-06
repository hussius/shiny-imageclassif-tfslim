#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Object recognition with pretrained model"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      fileInput('img', 'Choose file to upload',
                accept = c('.jpg',
                  '.jpeg',
                  '.JPG'
                )),
      tableOutput("classifs")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
       plotOutput("classifImage")
    )
  )
))
