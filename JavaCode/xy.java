package com.dji.RDP.JavaCode;

/**
 * saves the x,y positions of a point
 */
public class xy extends EV3Numeric {
    // Shows the point x and point y in a measurement of pixels (on picture)
    private double x;
    private double y;

    public xy(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    /**
     * calculates the distance between the instance xy(this) and another xy
     * @param p - xy position class
     * @return the distance as float
     */
    public int distance(xy p){return (int) Math.sqrt(Math.pow(this.x-p.getX(),2)+Math.pow(this.y-p.getY(),2));}

    /**
     * returns the relative direction of the next xy position class
     * @param p - xy position class
     * @return - left(1), front(2), right(3)
     * @throws Exception -
     */
    public int direction(xy p) throws Exception
    {
        int turn,ans;
        if(this.x==p.getX())
        {
            if(this.y>p.getY())
                turn=2;
            else
                turn=0;
        }
        else
        {
            if(this.x>p.getX())
                turn=1;
            else
                turn=3;
        }
        ans=(turn-direction)%4;
        direction=turn;
        //offsetting to make always positive
        if(ans<0)
            ans+=4;
        //front
        if(ans==0)
            return 2;
        //turn 180 degrees, not compatible with are program
        if(ans==2)
            throw new Exception();
        return ans;
    }
}