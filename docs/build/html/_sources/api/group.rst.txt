``farmer.group`` — Group
=========================

.. module:: farmer.group

A :class:`Group` represents one to five nearby sources modeled simultaneously
by The Tractor. Groups are created by :meth:`~farmer.brick.Brick.spawn_group`
and are normally short-lived objects — their results are absorbed back into
the parent brick after processing.

.. autoclass:: farmer.group.Group
   :members:
   :special-members: __init__
   :show-inheritance:
