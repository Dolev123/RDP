package com.dji.RDP.JavaCode;

import java.util.ArrayList;

public class Point {

    //Id that will be used to identify the point
    private String id;

    //Will show if we have visited said point or not
    private boolean visited;

    //All neighbors that are adjacent to said point
    private ArrayList<Point> neighbors;

    //minimum distance found for said point
    private double minDistance;

    //Predecessor of point from which the minimum Distance was found
    private Point pred;


    private double x, y;

    public Point(String id, double x, double y) {
        this.id = id;
        visited = false;
        this.pred = null;
        this.x = x;
        this.y = y;
        this.minDistance = Double.MAX_VALUE;
    }

    public Point(String id) {
        this.id = id;
        visited = false;
        this.pred = null;
        this.neighbors = new ArrayList<Point>();
        this.minDistance = Double.MAX_VALUE;
    }


    public Point getPred() {
        return pred;
    }


    public void setPred(Point pred) {
        this.pred = pred;
    }

    public String getId() {
        return id;
    }


    public void setId(String id) {
        this.id = id;
    }


    public boolean isVisited() {
        return visited;
    }


    public void setVisited(boolean visited) {
        this.visited = visited;
    }


    public ArrayList<Point> getNeighbors() {
        return neighbors;
    }


    public void setNeighbors(ArrayList<Point> neighbors) {
        this.neighbors = neighbors;
    }


    public double getMinDistance() {
        return minDistance;
    }


    public void setMinDistance(double minDistance) {
        this.minDistance = minDistance;
    }


    public double getX() {
        return this.x;
    }

    public void setX(double x) {
        this.x = x;
    }


    public double getY() {
        return this.y;
    }


    public void setY(double y) {
        this.y = y;
    }
}