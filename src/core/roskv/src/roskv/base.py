#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
API and some documentation stubs are liberally cribbed from cablehead/python-consul, MIT License
https://github.com/cablehead/python-consul
"""
from typing import Union, Tuple, List, Dict
from abc import abstractmethod, ABCMeta


class NullDefault(object):
    """A default placeholder for parameters that can take a meaningful None"""

    def __str__(self):
        return "<default>"


Jsonable = Union[dict, list, int, bool, str, tuple, float, NullDefault]

_default = NullDefault()


class Getter:
    """Interface for HTTP GET
    The GET method requests a representation of the specified resource. Requests using GET should only retrieve data.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, key, default=_default, **kwargs):
        # type: (str, Jsonable, Jsonable) -> Jsonable
        pass


class Header:
    """Interface for HTTP HEAD
    The HEAD method asks for a response identical to that of a GET request, but without the response body.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def head(self, key, default=_default, **kwargs):
        # type: (str, Jsonable, Jsonable) -> Union[dict, str]
        pass


class Putter:
    """Interface for HTTP PUT
    The PUT method replaces all current representations of the target resource with the request payload.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def put(self, key, val, **kwargs):
        # type: (str, Jsonable, Jsonable) -> Union[int, bool, str]
        pass


class Poster:
    """Interface for HTTP POST
    The POST method is used to submit an entity to the specified resource, often causing a change in state or side
    effects on the server.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def post(self, key, params, **kwargs):
        # type: (str, dict, Jsonable) -> Union[int, bool, str]
        pass


class Patcher:
    """Interface for HTTP PATCH
    The PATCH method is used to apply partial modifications to a resource.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def patch(self, key, params, **kwargs):
        # type: (str, dict, Jsonable) -> Union[int, bool, str]
        pass


class Deleter:
    """Interface for HTTP DELETE
    The DELETE method deletes the specified resource.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def delete(self, key, **kwargs):
        # type: (str, Jsonable) -> Union[int, bool, str]
        pass


class Monostack(object):
    """A stack with 0 or 1 objects"""

    def __init__(self):
        self._obj = None

    def push(self, obj):
        """Write some object to serve as the current output of the service"""
        self._obj = obj

    def pop(self):
        obj = self._obj
        self._obj = None
        return obj

    def peek(self):
        return self._obj

    def __bool__(self):
        return self._obj is not None

    def __len__(self):
        if self._obj is None:
            return 0
        return 1


class ChildService(Monostack):
    def __init__(self, kv):
        self.kv = kv
        super(ChildService, self).__init__()

    @abstractmethod
    def set_health(self, status):
        # type: (str) -> bool
        """Set the output health state"""

    def healthy(self):
        self.set_health("passing")

    def unhealthy(self):
        self.set_health("critical")


class KV(Getter, Putter, Deleter):
    """Interface for a bare-bones key-value store"""

    __metaclass__ = ABCMeta


class Health(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def service(
        self, service, index=None, wait=None, passing=None, tag=None, dc=None, near=None, token=None, node_meta=None
    ):
        # type: (str, int, str, bool, str, str, str, str, dict) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *nodes*)
        """
        pass

    @abstractmethod
    def checks(self, service, index=None, wait=None, dc=None, near=None, token=None, node_meta=None):
        # type: (str, int, str, str, str, str, dict) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *checks*) with *checks* being the
        checks associated with the service.
        """
        pass

    @abstractmethod
    def state(self, name, index=None, wait=None, dc=None, near=None, token=None, node_meta=None):
        # type: (str, int, str, str, str, str, dict) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *nodes*)
        """
        pass

    @abstractmethod
    def node(self, node, index=None, wait=None, dc=None, token=None):
        # type: (str, int, str, str, str) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *checks*)
        """
        pass


class Catalog(object):
    __metaclass__ = ABCMeta

    def register(self, node, address, service=None, check=None, dc=None, token=None):
        raise NotImplementedError()

    def deregister(self, node, service_id=None, check_id=None, dc=None, token=None):
        raise NotImplementedError()

    def datacenters(self):
        """
        Returns all the datacenters that are known by the Consul server.
        """
        raise NotImplementedError("this api is single-datacenter only")

    @abstractmethod
    def nodes(self, index=None, wait=None, consistency=None, dc=None, near=None, token=None, node_meta=None):
        """
        Returns a tuple of (*index*, *nodes*) of all nodes known
        about in the *dc* datacenter. *dc* defaults to the current
        datacenter of this agent.

        *index* is the current Consul index, suitable for making subsequent
        calls to wait for changes since this query was last run.
        Maps to /catalog/nodes
        """

    @abstractmethod
    def services(self, index=None, wait=None, consistency=None, dc=None, token=None, node_meta=None):
        """
        Returns a tuple of (*index*, *services*) of all services known
        about in the *dc* datacenter. *dc* defaults to the current
        datacenter of this agent.

        Maps to /catalog/services
        """

    def node(self, node, index=None, wait=None, consistency=None, dc=None, token=None):
        """
        Returns a tuple of (*index*, *services*) of all services provided
        by *node*.
        """
        raise NotImplementedError()


class Agent(object):
    """
            The Agent endpoints are used to interact with a local Consul agent.
            Usually, services and checks are registered with an agent, which then
            takes on the burden of registering with the Catalog and performing
            anti-entropy to recover from outages.
            """

    __metaclass__ = ABCMeta

    def self(self):
        """
        Returns configuration of the local agent and member information.
        """
        raise NotImplementedError()

    def services(self):
        """
        Returns all the services that are registered with the local agent.
        These services were either provided through configuration files, or
        added dynamically using the HTTP API. It is important to note that
        the services known by the agent may be different than those
        reported by the Catalog. This is usually due to changes being made
        while there is no leader elected. The agent performs active
        anti-entropy, so in most situations everything will be in sync
        within a few seconds.
        """
        raise NotImplementedError()

    def checks(self):
        """
        Returns all the checks that are registered with the local agent.
        """
        raise NotImplementedError()

    def members(self, wan=False):
        """
        Returns all the members that this agent currently sees. This may
        vary by agent, use the nodes api of Catalog to retrieve a cluster
        wide consistent view of members.

        For agents running in server mode, setting *wan* to *True* returns
        the list of WAN members instead of the LAN members which is
        default.
        """
        raise NotImplementedError()

    def maintenance(self, enable, reason=None):
        raise NotImplementedError("Maintenance mode not implemented")

    def join(self, address, wan=False):
        """
        This endpoint instructs the agent to attempt to connect to a
        given address.

        *address* is the ip to connect to.

        *wan* is either 'true' or 'false'. For agents running in server
        mode, 'true' causes the agent to attempt to join using the WAN
        pool. Default is 'false'.
        """
        raise NotImplementedError()

    def force_leave(self, node):
        """
        This endpoint instructs the agent to force a node into the left
        state. If a node fails unexpectedly, then it will be in a failed
        state. Once in the failed state, Consul will attempt to reconnect,
        and the services and checks belonging to that node will not be
        cleaned up. Forcing a node into the left state allows its old
        entries to be removed.

        *node* is the node to change state for.
        """
        raise NotImplementedError()


class Service(object):
    __metaclass__ = ABCMeta

    def register(
        self,
        name,
        service_id=None,
        address=None,
        port=None,
        tags=None,
        check=None,
        token=None,
        # *deprecated* use check parameter
        script=None,
        interval=None,
        ttl=None,
        http=None,
        timeout=None,
        enable_tag_override=False,
    ):
        """
        Add a new service to the local agent. There is more
        documentation on services
        `here <http://www.consul.io/docs/agent/services.html>`_.

        *name* is the name of the service.

        If the optional *service_id* is not provided it is set to
        *name*. You cannot have duplicate *service_id* entries per
        agent, so it may be necessary to provide one.

        *address* will default to the address of the agent if not
        provided.

        An optional health *check* can be created for this service is
        one of `Check.script`_, `Check.http`_, `Check.tcp`_,
        `Check.ttl`_ or `Check.docker`_.

        *token* is an optional `ACL token`_ to apply to this request.
        Note this call will return successful even if the token doesn't
        have permissions to register this service.

        *script*, *interval*, *ttl*, *http*, and *timeout* arguments
        are deprecated. use *check* instead.

        *enable_tag_override* is an optional bool that enable you
        to modify a service tags from servers(consul agent role server)
        Default is set to False.
        This option is only for >=v0.6.0 version on both agent and
        servers.
        for more information
        https://www.consul.io/docs/agent/services.html
        """
        raise NotImplementedError()

    def deregister(self, service_id):
        """
        Used to remove a service from the local agent. The agent will
        take care of deregistering the service with the Catalog. If
        there is an associated check, that is also deregistered.
        """
        raise NotImplementedError()

    def maintenance(self, service_id, enable, reason=None):
        """
        The service maintenance endpoint allows placing a given service
        into "maintenance mode".

        *service_id* is the id of the service that is to be targeted
        for maintenance.

        *enable* is either 'true' or 'false'. 'true' enables
        maintenance mode, 'false' disables maintenance mode.

        *reason* is an optional string. This is simply to aid human
        operators.
        """

        raise NotImplementedError()


class Check(object):
    __metaclass__ = ABCMeta

    def register(
        self,
        name,
        check=None,
        check_id=None,
        notes=None,
        service_id=None,
        token=None,
        script=None,
        interval=None,
        ttl=None,
        http=None,
        timeout=None,
    ):
        """
        Register a new check with the local agent. More documentation
        on checks can be found `here
        <http://www.consul.io/docs/agent/checks.html>`_.

        *name* is the name of the check.

        *check* is one of `Check.script`_, `Check.http`_, `Check.tcp`_
        `Check.ttl`_ or `Check.docker`_ and is required.

        If the optional *check_id* is not provided it is set to *name*.
        *check_id* must be unique for this agent.

        *notes* is not used by Consul, and is meant to be human
        readable.

        Optionally, a *service_id* can be specified to associate a
        registered check with an existing service.

        *token* is an optional `ACL token`_ to apply to this request.
        Note this call will return successful even if the token doesn't
        have permissions to register this check.

        *script*, *interval*, *ttl*, *http*, and *timeout* arguments
        are deprecated. use *check* instead.

        Returns *True* on success.
        """
        raise NotImplementedError()

    def deregister(self, check_id):
        """
        Remove a check from the local agent.
        """
        raise NotImplementedError()

    def ttl_pass(self, check_id, notes=None):
        """
        Mark a ttl based check as passing. Optional notes can be
        attached to describe the status of the check.
        """
        raise NotImplementedError()

    def ttl_fail(self, check_id, notes=None):
        """
        Mark a ttl based check as failing. Optional notes can be
        attached to describe why check is failing. The status of the
        check will be set to critical and the ttl clock will be reset.
        """
        raise NotImplementedError()

    def ttl_warn(self, check_id, notes=None):
        """
        Mark a ttl based check with warning. Optional notes can be
        attached to describe the warning. The status of the
        check will be set to warn and the ttl clock will be reset.
        """
        raise NotImplementedError()


class BaseEnvoy(KV):
    __metaclass__ = ABCMeta
