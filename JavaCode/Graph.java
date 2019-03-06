package com.dji.RDP.JavaCode;

import java.util.ArrayList;

public class Graph {

    //point used at start, startPoint minDistance must equal 0
    private Point startPoint;
    //target point for end
    private Point endPoint;
    //Array list of all the points
    private ArrayList<Point> points;

    private ArrayList<xy> minFin;


    public Graph(Point startPoint, Point endPoint, ArrayList<Point> p) {
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.points = p;
        this.minFin = Dijkstra(startPoint, endPoint);
    }

    //finds distance between two points
    public double findDistance(Point p1, Point p2) {
        return (Math.sqrt(Math.pow(p1.getX() - p2.getX(), 2) + Math.pow(p1.getY() - p2.getY(), 2)));
    }

    //sets distance for all neighbors and returns the neighbor with the minimum distance
    public Point setAndMinDistanceNeighbor(Point p) {
        ArrayList<Point> arrlist = p.getNeighbors();
        //returned point
        Point finP = new Point("id", Double.MAX_VALUE, Double.MAX_VALUE);
        //finP.setMinDistance(Double.MAX_VALUE-1);
        for (int counter = 0; counter < arrlist.size(); counter++) {
            //current point wasn't visited
            if (arrlist.get(counter).isVisited() == false) {
                double d = findDistance(arrlist.get(counter), p) + p.getMinDistance();
                if (arrlist.get(counter).getMinDistance() > d) {
                    arrlist.get(counter).setMinDistance(d);
                    arrlist.get(counter).setPred(p);
                }
            }
        }
        p.setVisited(true);
        System.out.println("Point " + p.getId().toString() + " visited " + p.isVisited());
        for (int i = 0; i < points.size(); i++) {
            if (points.get(i).isVisited() == false) {
                if (points.get(i).getMinDistance() < finP.getMinDistance()) {
                    finP = points.get(i);
                }
            }
        }
        return finP;
    }

    //receives two points to start and end with, and returns an array list with the points of the minimum distance
    public ArrayList<xy> Dijkstra(Point startPoint, Point endPoint) {
        startPoint.setMinDistance(0);
        startPoint.setPred(null);
        setMinDistance(startPoint, endPoint);
        ArrayList<xy> minFin = new ArrayList();
        ArrayList<xy> minFinr = new ArrayList();
        Point p = endPoint;
        do {
            minFinr.add(new xy(p.getX(), p.getY()));
            p = p.getPred();
        } while (p.getPred() != null);
        minFinr.add(new xy(p.getX(), p.getY()));
        for (int i = minFinr.size() - 1; i > -1; i--) {
            minFin.add(minFinr.get(i));
        }
        return minFin;
    }

    //sets all points to have the id before them with minimum distance
    public void setMinDistance(Point p, Point endPoint) {
        System.out.println(p.getId());
        Point p2 = setAndMinDistanceNeighbor(p);
        if (p2.getId().equals(endPoint.getId())) {
            return;
        } else {
            setMinDistance(p2, endPoint);
        }

    }

    public ArrayList<xy> getPoints() {
        return this.minFin;
    }

    public ArrayList<xy> getMinFin() {
        return minFin;
    }

    public void setMinFin(ArrayList<xy> minFin) {
        this.minFin = minFin;
    }


}


