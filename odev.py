import cv2
import matplotlib as plt 

#resmi gri tonlamalı olarak yükle
img = cv2.imread("animals.jpg",0)
#resmi göster
cv2.imshow("Resim",img)

#kenarları algıla
edges = cv2.Canny(img,100,200)

#çizimi göster
cv2.imshow("Cizilmis Resim",edges)

print(img.shape) #Resmin boyutu

new_width = int(img.shape[1] * 4/5) # Genişlik
new_height = int(img.shape[0] * 4/5)# Yükseklik
resized_img = cv2.resize(img,(new_width,new_height)) #yeniden boyutlandır
cv2.imshow("Yeniden Boyutlandirilmis(4/5) Resim",resized_img)

#Resme köpek yazısını ekle
font = cv2.FONT_HERSHEY_SIMPLEX #font tipi
text = 'Dog'
position = (265,45)
font_scale = 1
color= (0,255,0)
thickness = 2
#Metin ekle
cv2.putText(img, text, position,font,font_scale,color,thickness)
#Köpek yazısı eklenmiş resmi göster
cv2.imshow("Orijinal Resim - Kopek",img)    
#50 threshold değeri üzerindekileri beyaz altındakileri siyah yapalım
#binary threshold yöntemi kullanalım ve resmi çizdirelim
_, thresh_img = cv2.threshold(img, thresh = 50 , maxval =255, type = cv2.THRESH_BINARY) 
cv2.imshow('Threshold',thresh_img)
#orijinal resme gaussian bulanıklaştırma uygulayalım
gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
cv2.imshow('İmg with gb',gb)
#orijinal resme Laplacian gradyan uygulayalım ve resmi çizdirelim
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_64F)
cv2.imshow('Laplacian',laplacian)
#orijinal resmin histogramını çizdirelim


# Histogram hesaplama
img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Histogramı çizdirme
plt.figure()
plt.plot(img_hist.ravel())
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

