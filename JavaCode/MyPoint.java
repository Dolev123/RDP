package com.dji.RDP.JavaCode;

import java.util.Comparator;

/**
 * This class represents a detected point from the image. This class also contains a Color property, for
 * identifying the type of sticker( RED - Passable, BLUE - impassable, GREEN - end)
 */
public class MyPoint implements Comparator<MyPoint> {
    public double x;
    public double y;
    public Color color;

    public MyPoint() {
        this.x = this.y = 0;
        this.color = Color.NONE;
    }

    public MyPoint(double x, double y, Color color) {
        this.x = x;
        this.y = y;
        this.color = color;
    }

    @Override
    public int compare(MyPoint p1, MyPoint p2) {
        //=================Important to adapt thresh according to height of picture made by drone=============
        int thresh = 80;
        int xComp = Double.compare(p1.x, p2.x);
        //if they are on the same column, check who is the higher one.
        if (Math.abs(p1.x - p2.x) <= thresh) {
            return -Double.compare(p1.y, p2.y);
        } else
            return xComp;
    }

}

enum Color {
    RED, GREEN, BLUE, NONE
}