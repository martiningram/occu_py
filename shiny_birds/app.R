library(shiny)
library(raster)
library(gridExtra)
library(purrr)
library(ggplot2)

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
      
      selectInput(inputId = "species",
                  label = "Species to plot",
                  choices = combined_names,
                  selected = 'Barred Owl (Strix varia)'),
    h4('Select a species to view a comparison of five range maps:'),
    p('1.: BBS PA: A presence-absence (PA) model fitted to data from the North American Breeding bird survey (BBS) [2]'),
    p('2.: eBird PA: A presence-absence (PA) model fitted to data from eBird [3]'), 
    p('3.: BBS MSOD: A multi-species occupancy detection (MSOD) model fitted to BBS [2]'),
    p('4.: eBird MSOD: A multi-species occupancy detection (MSOD) model fitted to eBird [3]'),    
    p('5.: BirdLife expert map: An expert map from BirdLife International (see citation [1] below)'),
    p('In addition, a bar chart shows the discrepancy between each of these range maps and the expert map. Note that the expert map may not be flawless, so a larger error does not necessarily imply a worse prediction.'),
    p('[1] BirdLife International and Handbook of the Birds of the World (2020) Bird species distribution maps of the world. Version 2020.1. Available at http://datazone.birdlife.org/species/requestdis.'),
    p('[2] Pardieck, K.L., Ziolkowski Jr., D.J., Lutmerding, M., Aponte, V.I., and Hudson, M-A.R., 2020, North American Breeding Bird Survey Dataset 1966 - 2019: U.S. Geological Survey data release, https://doi.org/10.5066/P9J6QUF6.'), 
    p('[3] eBird. 2021. eBird: An online database of bird distribution and abundance [web application]. eBird, Cornell Lab of Ornithology, Ithaca, New York. Available: http://www.ebird.org.'),
    h4('Some species that may be interesting:'),
    p('Select the Spotted Sandpiper (Actitis macularius) for a map which eBird MSOD matches best'),
    p('Select the Eastern Screech-Owl (Megascops asio) for a map which BBS MSOD matches best')
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      align='center',
      
      plotOutput(outputId = "predictions", height = 500),
      plotOutput(outputId = 'range_map', height=200),
      plotOutput(outputId = 'agreement', height=300, width=600)
      
    ),
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output) {
  
  output$range_map <- renderPlot({
    
    cur_species <- scientific_name_from_combined(input$species)
    rasters <- get_rasters(base_dirs, cur_species)
    
    plot_raster_on_us(rasters$expert, title = 'BirdLife expert map', zero_one_probs=TRUE)
    
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
  
  output$agreement <- renderPlot({
    
    cur_species <- scientific_name_from_combined(input$species)
    rasters <- get_rasters(base_dirs, cur_species)
    base_raster <- rasters$expert
    differences <- lapply(rasters, function(x) cellStats((x - base_raster)^2, 'mean'))
    
    to_plot <- data.frame(model=names(differences), brier=unlist(differences))
    to_plot <- to_plot[to_plot$model != 'expert', ]
    
    renamings <- c(
      'bbs_pa'="BBS PA",
      'ebird_pa'="eBird PA",
      'bbs_occu'="BBS MSOD",
      'ebird_occu'='eBird MSOD'
    )
    
    to_plot$model <- renamings[to_plot$model]
    
    ggplot(to_plot, aes(x=model, y=brier, fill=model)) +
      geom_bar(stat='identity') +
      theme_minimal() +
      ggtitle('Average error compared to expert map (lower = closer agreement)') +
      # ylim(0, 1) +
      xlab('Model') +
      ylab('Brier score (average mean square error)') +
      theme(legend.position = 'none')
    
  })
  
}

shinyApp(ui = ui, server = server)
