library(maps)
library(maptools)
library(mapview)
library(rasterVis)

plot_raster_on_us <- function(raster, title = '', zero_one_probs=FALSE) {

  # Weirdly, may have to run these two outside the function.
  countries <- maps::map("state", plot=FALSE) 
  countries <- maptools::map2SpatialLines(countries, proj4string = CRS("+proj=longlat"))
  
  myPal <- RColorBrewer::brewer.pal('Blues', n=9)
  myTheme <- rasterTheme(region = myPal)
  
  ## Overlay lines on levelplot
    if (zero_one_probs) {
        p <- levelplot(raster, margin=FALSE, par.settings = myTheme, main = title,
                    at = seq(0, 1, length.out = 100),
                    cuts= 100, scales=list(x=list(draw=FALSE), y=list(draw=FALSE)), xlab='', ylab='',
                    axis.line = list(col='transparent')) +
            layer(sp.lines(countries, fill='black', col='black'), data=list(countries=countries))
    } else {
        p <- levelplot(raster, margin=FALSE, par.settings = myTheme, main = title,
                       cuts=100,
                    scales=list(x=list(draw=FALSE), y=list(draw=FALSE)), xlab='', ylab='',
                    axis.line = list(col='transparent')) +
            layer(sp.lines(countries, fill='black', col='black'),
                  data=list(countries=countries))
    }
  
  p
}
