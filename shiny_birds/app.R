library(shiny)
library(raster)
library(gridExtra)
library(purrr)

source('./map_util.R', local = TRUE)

base_dirs = c(
    ebird_pa='./ebird_pa/',
    bbs_pa='./bbs_pa/',
    ebird_occu='./ebird_occu/',
    bbs_occu='./bbs_occu/',
    expert='./expert/'
)

species_names <- 
  base_dirs %>%
  purrr::map(function(x) Sys.glob(paste0(x, '*.grd'))) %>%
  purrr::map(basename) %>%
  purrr::map(tools::file_path_sans_ext)

shared_species <- reduce(species_names, intersect)

# Find the common names
lookup <- read.csv('./species_info_2019.csv', row.names = 1)
common_names <- lookup$common_name
names(common_names) <- lookup$scientific_name
common_name_lookup <- common_names[shared_species]

to_combined_name <- function(common_name, scientific_name) {
  paste0(common_name, ' (', scientific_name, ')')
}

scientific_name_from_combined <- function(combined_name) {
  # Credit to: https://stackoverflow.com/questions/13498843/regex-to-pickout-some-text-between-parenthesis
  gsub(".*\\((.*)\\).*", "\\1", combined_name)
  
}

combined_names <- to_combined_name(as.character(common_name_lookup), 
                                   names(common_name_lookup))

combined_names <- sort(combined_names)

get_rasters <- function(base_dirs, species_name) {
  
    paths <- paste0(base_dirs, '/', species_name, '.grd')
    loaded <- lapply(paths, raster)
    names(loaded) <- names(base_dirs)  
    
    loaded
    
}

# Define UI for app that draws a histogram ----
ui <- fluidPage(
  
  # App title ----
  titlePanel("Range map comparisons"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of bins ----
      selectInput(inputId = "species",
                  label = "Species to plot",
                  choices = combined_names,
                  selected = 'Barred Owl (Strix varia)')
      
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      plotOutput(outputId = "predictions", height = 600),
      plotOutput(outputId = 'range_map', height=400)
      
    )
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output) {
  
  output$range_map <- renderPlot({
    
    cur_species <- scientific_name_from_combined(input$species)
    rasters <- get_rasters(base_dirs, cur_species)
    
    plot_raster_on_us(rasters$expert, title = 'IUCN expert map')
    
  })
  
  output$predictions <- renderPlot({
    
    cur_species <- scientific_name_from_combined(input$species)
    rasters <- get_rasters(base_dirs, cur_species)
    
    plot_a <- plot_raster_on_us(rasters$bbs_pa, title = 'BBS PA')
    plot_b <- plot_raster_on_us(rasters$ebird_pa, title = 'eBird PA')
    plot_c <- plot_raster_on_us(rasters$bbs_occu, title = 'BBS MSOD')
    plot_d <- plot_raster_on_us(rasters$ebird_occu, title = 'eBird MSOD')

    grid.arrange(plot_a, plot_b, plot_c, plot_d, ncol=2) 
    
  })
  
}

shinyApp(ui = ui, server = server)
