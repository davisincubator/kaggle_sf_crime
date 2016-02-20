#####--------------------------#####
#####   Kaggle SF crime        #####
#####--------------------------#####

dir()
dat <- read.csv("train.csv")
head(dat)
table(dat$Category)
which(dat$Category == "TREA")
dat[which(dat$Category == "ASSAULT"), ]$Descript


######################################################
# plot of number of crimes in each category per year #
######################################################

year <- strsplit(as.character(dat$Dates), "-")
head(year)
y <- sapply(1:length(year), function (x) year[[x]][1])
head(y)
dat$year <- y
class(dat$Dates)
length(year)
table(dat$year)

# checking for missing data
sum(is.na(dat$Dates))

crime_table <- dat %>%
  group_by(Category) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# grouping by crime
library(dplyr)
crime_cat <- dat %>% 
  group_by(year, DayOfWeek, Category) %>%    
  summarise(count = n()) %>%
  arrange(desc(count))

# plot of freq crime cats by year and day of week  
library()
ggplot(aes(x = DayOfWeek, y = count, fill = Category), data = crime_cat) +
  geom_bar(stat = "identity", position = "Dodge") + facet_grid(year ~ .)

# non faceted plot of freq of crime cats by year
ggplot(aes(x = year, y = count, fill = Category), data = crime_cat) +
  geom_bar(stat = "identity", position = "Dodge") 

# plot of frequency of crime categories by year
ggplot(aes(x = Category, y = count), data = crime_cat) +
  geom_bar(stat = "identity") + facet_grid(year ~ .) +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# grouping by district
names(dat)
crime_cat_dist <- dat %>% 
  group_by(year, PdDistrict, Category) %>%    
  summarise(count = n()) %>%
  arrange(desc(count))

ggplot(aes(x = PdDistrict, y = count, fill = Category), data = crime_cat_dist) +
  geom_bar(stat = "identity", position = "Dodge") + facet_grid(year ~ .)

# grouping by district and not year
names(dat)
crime_cat_dist2 <- dat %>% 
  group_by(PdDistrict, Category) %>%    
  summarise(count = n()) %>%
  arrange(desc(count))


# messing around with longitude and latitude

smoothScatter(dat$X, dat$Y, xlim = c(-122.5, -122.35))
d <- densCols(dat$X, dat$Y, colramp = colorRampPalette(rev(rainbow(10, end = 4/6))))
p <- ggplot(df) +
  geom_point(aes(x, y, col = d), size = 1) +
  scale_color_identity() +
  theme_bw()
print(p)


# playing with ggmap function
library(ggmap)
sf <- "the castro"
qmap(sf, zoom = 13, source = "stamen", maptype = "toner")
sf_map <- qmap(sf, zoom = 13)

sf_map +
geom_point(aes(x = X, y = Y, colour = Category),
           data = dat)


# subsetting by violent crimes: robery, assault, sex offenses forcible
# and creating ggmap plot with crimes by location

vio_crimes <- subset(dat, Category == "ROBBERY" | Category == "ASSAULT" | 
                      Category == "SEX OFFENSES FORCIBLE") 

index <- dat$Category %in% c("ROBBERY", "ASSAULT", "SEX OFFENSES FORCIBLE")
d <- dat[index, ]

sf <- "the castro"
sf_map <- qmap(sf, zoom = 13)
sf_map +
  geom_point(aes(x = X, y = Y, colour = Category),
             data = vio_crimes)
