# Calculus Study Guide

## Sources:
  Problems: Calculus - Gilbert Strang
  Answers: Calculus: Chapter 1 Study Guide - Gilbert Strang

---
## Chapter 1 - Introduction to Calculus

### Section 1 - Velocity and Distance
#### Problems
  10. Draw any reasonable graphs of v(t) and f(t) when:

    a. The driver backs up, stops to shift gear, then goes fast
      f(t): negative at a constant slope,
            stop and stay at that level,
            positive at a faster constant slope
      v(t): negative, zero, positive
    b. The driver slows to 55 for a police car
      f(t): positive at a constant slope,
            positive at a slower rate
      v(t): above 55, equal to 55
    c. In a rough gear change, the car accelerates in jumps
      f(t): positive constant slope that increases at every velocity jump
      v(t): increases in jumps
    d. The driver waits for a light that turns green
      f(t): zero, then constant positive slope
      v(t): zero, then positive

  26. When f(t) = vt + C, find the constants v and C, where f(t+1) = f(t)+2.  

      f(t+1) = f(t)+2
      v(t+1) + C = vt + C + 2
      vt + v + C = vt + C + 2
      v + C = C + 2
      v = 2

      f(t) = 2t + C, where C = f(0)

  36. Ten hours after the accident the alcohol reading was .061. Blood alcohol
    is eliminated at .015 per hour. What was the reading at the time of the
    accident? How much later would it drop to .04?

      f(10) = .061
      v(t) = -.015

      f(0) = ?
      f(0) = .061 + .015 * 10 = .211

      f(t) = .04, t = ?
      .061 - .04 = .021
      t = 10 + .021/.015 hours

  51. If f(t) = 3t-1 for 0 ≤ t ≤ 2, give formulas with domain and find the
    the slopes of these six functions:

      a. f(t+2):
        formula:  f(t+2) = 3(t+2)-1 = 3t+5
        domain:   0 ≤ t+2 ≤ 2 => -2 ≤ t ≤ 0
        slope:    3

      b. f(t)+2:
        formula:  f(t)+2 = 3t-1+2 = 3t+1
        domain:   0 ≤ t ≤ 2
        slope:    3

      c. 2f(t):
        formula:  2f(t) = 2(3t-1) = 6t-2
        domain:   0 ≤ t ≤ 2
        slope:    6

      d. f(2t):
        formula:  f(2t) = 3(2t)-1 = 6t-1
        domain:   0 ≤ 2t ≤ 2 => 0 ≤ t ≤ 1
        slope:    6

      e. f(-t):
        formula:  f(-t) = 3(-t)-1 = -3t-1
        domain:   0 ≤ -t ≤ 2 => -2 ≤ t ≤ 0
        slope:    -3

      f. f(f(t)):
        formula:  f(f(t)) = 3(3t-1)-1 = 9t-3-1 = 9t-4
        domain:   0 ≤ 3t-1 ≤ 2 => 1 ≤ 3t ≤ 3 => 1/3 ≤ t ≤ 1
        slope:    9

### Section 2 - Calculus without Limits
#### Problems
  8. Suppose v(t) = 10 for t < 1/10, v(t) = 0 for t > 1/10. Starting from
    f(0) = 1, find f(t) in two pieces.

      f(t) = 10t + 1 for 0 ≤ t ≤ 1/10
      f(t) = 2 for 1/10 ≤ t

  10. Suppose g(t) = 2t + 1 and f(t) = 4t. What are f(3) and g(f(3)) and
    g(f(t))? When t is changed to 4t, how much faster do distances increase?
    How much larger is velocity?

      f(3) = 4 * 3 = 12
      g(f(3)) = g(12) = 2 * 12 + 1 = 25
      g(f(t)) = g(4t) = 2 * 4t + 1 = 8t + 1

      Distance increases 4 times faster
      Velocity is 4 times bigger

  28. Given numbers f0, f1, f2... and their differences v_j = f_j - f_j-1,
    suppose the v's increase by 4 at every step. Show by example and then by
    algebra that the second difference f_j+1 - 2f_j + f_j-1 = 4

      If j = 0, f_j = 0, and v_j = 0, then:
        f_j+1 = 4, v_j+1 = 4
      Therefore:
        f_j+1 - 2f_j + f_j-1 = 4 - 2(0) + 0 = 4

      Furthermore:
        f_j+1 - 2f_j + f_j-1 = 4
        (f_j+1 - f_j) - (f_j - f_j-1) = 4
        v_j+1 - v_j = 4

  32. If the sequence v1, v2... has period 6 and w1, w2... has period 10, what
    is the period of v1 + w2, v2 + w2...?

      The period of v+w is 30, the smallest multiple of 6 and 10. Then v
      completes 5 cycles while w completes 3. An example for functions is:
      v = sin(πx/3) and w = sin(πx/5)

### Section 3 - The Velocity at an Instant
#### Problems
  2. For the following functions, compute [f(t+h) - f(t)]/h. This depends on t
    and h. Find the limit as h -> 0.

    a. f(t) = 6t

      [f(t+h) - f(t)]/h = [6(t+h) - 6t]/h = (6t + 6h - 6t)/h = 6h/h = 6

      limit as h -> 0: 6

    c. f(t) = 1/2 at^2

      [f(t+h) - f(t)]/h = [1/2 a (t+h)^2 - 1/2 at^2]/h
        = [1/2 a (t^2 + 2th + h^2) - 1/2 at^2]/h
        = [1/2 at^2 + ath + 1/2 ah^2 - 1/2 at^2]/h
        = (ath + 1/2 ah^2)/h = [h(at + 1/2 ah)]/h = at + 1/2 ah = a(t + 1/2h)

      limit as h -> 0: at

    e. f(t) = 6

      [f(t+h) - f(t)]/h = (6-6)/h = 0

      limit as h -> 0: 0

    f. v(t) = 2t

      f(t) = t^2

      [f(t+h) - f(t)]/h = ((t+h)^2 - t^2)/h = ((t^2 + 2th + h^2) - t^2)/h
        = (t^2 + 2th + h^2 - t^2)/h = (2th + h^2)/h = (h(2t + h))/h = 2t + h

      limit as h -> 0: 2t

  14. State true or false for any distance curves.

    a. The slope of the line from A to B is the average velocity between those
      points

      True, the slope is ∆f/∆t

    b. Secant lines have smaller slopes than the curve.

      False, the curve is partly steeper and partly flatter than the secant line

    c. If f(t) and F(t) start together and finish together, the average
      velocities are equal.

      True, because ∆f = ∆F

    d. If v(t) and V(t) start together and finish together, the increases in
      distance are equal.

      False, V could be larger in between

  18. Suppose v(t) = t for t ≤ 2 and v(t) = 2 for t ≥ 2. Draw the graph of f(t)
    out to t = 3.

      The graph is a parabola f(t) = 1/2t^2 out to f = 2 at t = 2. After that,
      the slope of f stays constant at 2.

  20. Suppose v(t) is the piecewise linear sine function of Section 1.2,
    Figure 1.9. Find the area under v(t) between t = 0 and t = 1, 2, 3, 4, 5, 6.
    Plot those points f(1),... f(6) and draw the complete piecewise parabola
    f(t).

      area(t = 0..1) = 1/2
      area(t = 0..2) = 3/2
      area(t = 0..3) = 2
      area(t = 0..4) = 3/2
      area(t = 0..5) = 1/2
      area(t = 0..6) = 0

      The graph of f(t) through these points is a symmetric
      parabola-line-parabola-line-parabola to zero.

### Section 4 - Circular Motion
#### Problems
  14. A mass falls from the top of the unit circle when the ball of speed 1
    passes by. What acceleration a is necessary to meet the ball at the bottom?

      The ball goes halfway around the circle in time π. For the mass to fall a
      distance 2 in time π, we need:
        2 = 1/2 aπ^2, so a = 4/π^2

  18. Find the area under v = cos(t) from the change in f = sin(t) from t = π/2
    to t = 3π/2

      The area is still f(t) = sin t, and sin 3π/2 - sin π/2 = -1 - 1 = -2

  20. The distance curve f = 2 cos 3t yields the velocity curve v = -6 sin 3t.
    Explain the -6.

      The radius is 2 and time is speeded up by 3, so the speed is 6. There is
      a minus sign because the cosine starts downward (ball moving left).

  26. The ball at x = cos t, y = sin t circles (1) counterclockwise (2) with
    radius 1 (3) starting from x = 1, y = 0, (4) at speed 1. Find (1)(2)(3)(4)
    for the following motion: x = 3 cos 4t, y = 3 sin 4t

      Counterclockwise with radius 3, starting at (3,0) with speed 12

### Section 5 - Review of Trigonometry
  14. From the formula for sin(2t + t), find sin 3t in terms of sin t.

      sin 3t = sin(2t+t) = sin 2t cos t + cos 2t sin t
        = (2 sin t cos t) cos t + (cos^2 t - sin^2 t) sin t
        = 3 sin t - 4 sin^3 t

  26. Find every Θ that satisfies the equation: sin Θ = Θ

      Θ = 0

  30. Match a sin x + b cos x with A sin(x + Φ). From equation 9, show that
    a = A cos Φ and b = A sin Φ. Square and add to find A. Divide to find
    tan Φ = b/a.

      A sin(x + Φ) = A sin x cos Φ + A cos x sin Φ, therefore
      a = A cos Φ, b = A sin Φ, then
      a^2 + b^2 = A^2 cos^2 Φ + A^2 sin^2 Φ = A^2, finally
      A = √(a^2 + b^2) and tan Φ = A sin Φ / A cos Φ = b/a

  34. Draw a graph of the following equation: y = 2 sin πx

      The amplitude and period of 2 sin πx are both 2
