package com.dji.RDP.JavaCode;

import java.util.ArrayList;

public class Circle {
    private int id;
    private double x;
    private double y;
    private ArrayList<Integer> neighborID;


    public ArrayList<Integer> getNeighborID() {
        return this.neighborID;
    }

    public void setNeighborID(ArrayList<Integer> neighborID) {
        this.neighborID = neighborID;
    }

    public int getId() {
        return this.id;
    }

    public void setId(int id) {
        this.id = id;
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

    public Circle(int id, double x, double y) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.neighborID = new ArrayList<Integer>();
    }
    public Circle(int id, double x, double y, ArrayList<Integer> neighbors) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.neighborID = neighbors;
    }

    public void findNeighbors(ArrayList<Circle> arraylist) {
        //start with vertical neighbors

        int size = arraylist.size();
        for (int i = 0; i < size - 1; i++) {
            if (arraylist.get(i).y < arraylist.get(i + 1).y) {
                arraylist.get(i).getNeighborID().add(i + 1);
                arraylist.get(i + 1).getNeighborID().add(i);
            }

        }
        //find size of column
        int horizsize = 1;
        for (int i = 0; i < size - 1; i++) {
            if (arraylist.get(i).y < arraylist.get(i + 1).y) {
                horizsize++;
            } else {
                i = size;
            }

        }
        //horizontal neighbors
        for (int i = 0; i < size - 1; i++) {
            if (horizsize + i < size) {
                if (arraylist.get(i).x < arraylist.get(i + horizsize).x) {
                    arraylist.get(i).getNeighborID().add(i + horizsize);
                }
                arraylist.get(i + horizsize).getNeighborID().add(i);
            }

        }

    }

    public ArrayList<Point> circleToPoint(ArrayList<Circle> circles) {
        ArrayList<Point> points = new ArrayList<Point>();
        for (int i = 0; i < circles.size(); i++) {
            Point a = new Point("" + circles.get(i).getId());
            a.setX(circles.get(i).getX() / 3.662);
            a.setY(circles.get(i).getY() / 3.662);
            points.add(a);
        }
        //put neighbors
        for (int i = 0; i < circles.size(); i++) {
            Point b = findPointByCircle(circles.get(i), points);
            ArrayList<Point> n = new ArrayList<Point>();
            for (int j = 0; j < circles.get(i).getNeighborID().size(); j++) {
                n.add(findPointByID(circles.get(i).getNeighborID().get(j), points));
            }
            b.setNeighbors(n);
        }
        return points;
    }

    public Point findPointByCircle(Circle circle, ArrayList<Point> p) {
        for (int i = 0; i < p.size(); i++) {
            if (p.get(i).getId().equals("" + circle.getId())) {
                return p.get(i);
            }
        }
        return p.get(0);
    }

    public Point findPointByID(int ID, ArrayList<Point> p) {
        for (int i = 0; i < p.size(); i++) {
            if (p.get(i).getId().equals("" + ID)) {
                return p.get(i);
            }
        }
        return p.get(0);

    }
}

