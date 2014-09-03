# -*- coding: utf-8 -*-
__author__ = 'tomek'
#/usr/bin/env python

'''

gray histogram

'''

import cv2
import numpy as np

def gray(img):
    '''
        histogram skali szarosci
    '''
    h = np.zeros((300, 256, 3))
    bins = np.arange(256).reshape(256, 1)

    hist_item = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    pts = np.column_stack((bins, hist))
    cv2.polylines(h, [pts], False, (255, 255, 255))

    h = np.flipud(h)

    cv2.imshow('grayhist', h)
    cv2.waitKey(0)


def color(img):
    '''
        histogram kolorow - niesprawdzone ponownie
    '''
    h = np.zeros((300, 256, 3))
    bins = np.arange(256).reshape(256, 1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h = np.flipud(h)

    cv2.imshow('colorhist', h)
    cv2.waitKey(0)


def length(peri):
    u'''
    pokazuje histoggram dkonturow na podstawie podanych długości konturów
    '''
    h = np.zeros((400, 101, 3))
    bins = range(101)
    bins = [int(bin * max(peri) / 100) for bin in bins]
    hist = [0] * 101
    for i in range(1, 101):
        hist[i] = len([1 for elem in peri if bins[i - 1] < elem < bins[i]])
    pts = np.column_stack((range(101), [2 * hi for hi in hist]))
    cv2.polylines(h, [pts], False, (255, 255, 255))
    h = np.flipud(h)
    cv2.imshow('hist', h)
    cv2.waitKey(0)
    return


def draw(peri):
    u'''
    pokazuje histoggram ilości wystąpień
    '''
    ln = len(peri)
    h = np.zeros((400, ln, 3))

    pts = np.column_stack((range(ln), [2 * hi for hi in peri]))

    cv2.polylines(h, [pts], False, (255, 255, 255))
    h = np.flipud(h)
    return h

def draw2(peri):
    u'''
    pokazuje histoggram ilości wystąpień
    '''
    ln = len(peri)
    h = np.zeros((1000, ln, 3))
    # h[:,:] = (255,255,255)

    avg = np.average(peri)
    peri2 = peri
    pts2 = np.column_stack((range(ln), [hi for hi in peri2]))

    peri = np.convolve(peri,[1,1,1,1,1],'same')
    peri = peri.tolist()
    peri = [int(p) for p in peri]

    # poly = np.polyfit(pts[:,0],pts[:,1],20)
    # val = np.polyval(poly,pts[:,0])
    # val = np.array(val, 'uint8')
    avg = int(avg)*3*5

    h[avg,range(ln)] = (0,0,255)

    # pogrubienie lini średniej
    # h[avg+1,range(ln)] = (0,0,255)
    # h[avg-1,range(ln)] = (0,0,255)

    # całka
    integ = [int(np.trapz(peri2[0:i])) for i in range(ln)]
    # differ = np.diff(integ)
    # strome = np.where(differ > 100)
    pts2 = np.column_stack((range(ln), [int(itg/25) for itg in integ]))

    #miejsca przeciec
    line = np.asarray(peri)
    line = line - avg
    s = np.sign(line)
    d = np.diff(s)
    crossings = np.nonzero(d)
    dist = np.diff(crossings)
    peaks = [dist[0][n] for n in range(len(dist[0])) if n%2 == 0]
    theWidestStart = np.argmax(peaks) * 2
    theWidestEnd = theWidestStart+1
    # wspolrzedne miejsca przeciecia najgrubszego piku
    a = crossings[0][theWidestStart]
    b = crossings[0][theWidestEnd]
    # pogrubienie lini
    # periNP = [0]
    # periNP.extend(peri)
    # periNP = periNP[0:-1]
    # pts3 = np.column_stack((range(ln), [hi for hi in periNP]))

    # line = line.clip(avg)
    pts = np.column_stack((range(ln), [hi for hi in peri]))

    # differ = [int(np.diff(peri2[0:i])) for i in range(ln)]
    # peri3 = [int(itg/25) for itg in integ]
    # differ = np.diff(peri3)
    # differ = np.append(differ,differ[-1])
    # pts3 = np.column_stack((range(ln), [int(itg/25) for itg in differ]))

    cv2.polylines(h, [pts2], False, (255, 255, 255))
    cv2.polylines(h, [pts], False, (255, 255, 255))
    # cv2.polylines(h, [pts3], False, (0, 0, 0))

    h[:,crossings] = (255,0,255)

    # pogrubienie lini średnich
    # h[:,crossings[0]+1] = (255,0,255)
    # h[:,crossings[0]-1] = (255,0,255)

    # cv2.polylines(h, [pts3], False, (0, 255, 255))
    h = np.flipud(h)
    return h,a,b