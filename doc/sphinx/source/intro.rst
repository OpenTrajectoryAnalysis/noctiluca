Quickstart
==========

`noctiluca` is a python library for downstream analysis of single particle
tracking data. Downstream here means that we are not concerned with particle
detection from movies or the linking problem, instead our starting point are
the linked particle trajectories. Consequently, the core of this library are
the classes `Trajectory <noctiluca.trajectory.Trajectory>`, representing a
single trajectory, and `TaggedSet <noctiluca.taggedset.TaggedSet>`, which
provides a useful way of organizing multiple (sets of) trajectories. In
addition, some utility code is provided to a) directly perform simple analyses
and b) facilitate the developement of analysis libraries building on the format
introduced here (such as `bayesmsd
<https://github.com/OpenTrajectoryAnalysis/bayesmsd>`_ and `bild
<https://github.com/OpenTrajectoryAnalysis/bild>`_).

So what's the fastest way to get started?

* Take a look at the tutorials for :doc:`Trajectory <examples/01_Trajectory>`
  and :doc:`TaggedSet <examples/02_TaggedSet>`
* For any details that are skipped in the tutorials, refer to the
  :doc:`documentation <noctiluca>`
* To see more of what `noctiluca` can do (and some standard usage), work
  through the rest of the :doc:`examples <examples>`
