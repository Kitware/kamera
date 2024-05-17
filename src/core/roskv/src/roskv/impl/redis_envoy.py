#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from typing import Any, Union, List, Mapping, Tuple, Dict
import sys
import json
import time

from six import string_types, binary_type
import redis
from benedict import benedict
from roskv.base import (
    BaseEnvoy,
    KV,
    Health,
    NullDefault,
    Jsonable,
    Agent,
    Catalog,
    Check,
    Service,
    ChildService,
)
from roskv.util import redis_decode, redis_encode, loader23
from vprint import vprint

_default = NullDefault()


class RedisKV(redis.client.Redis, KV):
    """Redis-based key-value store"""

    def __init__(self, agent_name, cfgdict):
        self.agent_name = agent_name
        super(RedisKV, self).__init__(**cfgdict)

    def get(self, key, default=_default, **kwargs):
        # type: (str, Jsonable, Any) -> Jsonable
        """
        Get a single value from a keypath.

        Automatically tries to deserialize from json, skipping on failure.

        :param key: key
        :param default: Default value if key not found
        :param kwargs: Not currently used
        :return: data dict at key
        """
        maybe_keys = self.keys(key)
        if len(maybe_keys) == 1:
            val = super(RedisKV, self).get(key, **kwargs)
            return loader23(val)
        return self.get_dict(key, default=default, **kwargs)

    def get_dict(self, key, default=_default, flatten=False, **kwargs):
        # type: (str, Jsonable, bool, Any) -> Jsonable
        """
        Get a nested value from a keypath.

        This may be deprecated in favor of overloading `get`

        Automatically tries to deserialize from json, skipping on failure.

        :param key: key or keypath prefix
        :param default: Not currently implemented
        :param kwargs: Not currently used
        :return: data dict at keypath
        """
        # key = key.strip("/")
        keys = self.keys(key + "*")
        if not keys:
            if default is _default:
                raise KeyError(key)
            else:
                return default
        p = self.pipeline()
        for k in keys:
            p.get(k)

        vals = p.execute()
        as_json = kwargs.get("as_json", True)
        return redis_decode(keys, vals, key, flatten, as_json)

    def put(self, key, val, **kwargs):
        # type: (str, Union[Any, Dict], Any) -> bool
        """
        Insert value into KV. If the val is a dict, automatically flatten and insert the keypath/value sequence.
        Automatically tries to serialize acceptable types to json.

        Consul's put interface returns True on success. Not sure best way to handle it so just returning as is.
        :param key: key or keypath prefix
        :param val: Singular value or dict
        :param kwargs: Not currently used
        :return: success of transaction
        """

        tups = redis_encode(key, val)
        if len(tups) == 0:
            return self.set(tups[0][0], tups[0][1], **kwargs)

        p = self.pipeline()
        for pair in tups:
            p.set(pair[0], pair[1])

        res = p.execute()
        return res

    def _put(self, key, val, **kwargs):
        if not isinstance(val, (string_types, binary_type)):
            val = json.dumps(val)

        return self.set(key, val, **kwargs)

    def delete(self, key, **kwargs):
        # key = key.strip("/")
        return super(RedisKV, self).delete(key, **kwargs)

    def delete_dict(self, key, **kwargs):
        vprint("deleting {}".format(key))
        # key = key.strip("/")
        keys = self.keys(key + "*")
        if not keys:
            raise KeyError(key)

        p = self.pipeline()
        for k in keys:
            p.delete(k)

        vals = p.execute()
        return vals


class RedisHealth(Health):
    """Redis-based health checks"""

    def __init__(self, agent_name, kv):
        # type: (str, RedisEnvoy) -> None
        self.agent_name = agent_name
        self.kv = kv

    def __del__(self):
        vprint("health object destroyed")

    def service(
        self, service, index=None, wait=None, passing=None, tag=None, dc=None, near=None, token=None, node_meta=None,
    ):
        # type: (str, int, str, bool, str, str, str, str, dict) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *nodes*)
        """
        pass

    def checks(
        self, service, index=None, wait=None, dc=None, near=None, token=None, node_meta=None,
    ):
        # type: (str, int, str, str, str, str, dict) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *checks*) with *checks* being the
        checks associated with the service.
        """
        pass

    def state(
        self, name, index=None, wait=None, dc=None, near=None, token=None, node_meta=None,
    ):
        # type: (str, int, str, str, str, str, dict) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *nodes*)
        """
        pass

    def node(self, node, index=None, wait=None, dc=None, token=None):
        # type: (str, int, str, str, str) -> Tuple[int, List[Dict]]
        """
        Returns a tuple of (*index*, *checks*)
        """
        pass


class RedisAgent(Agent):
    """Corresponds to the local Envoy agent.
    Usually, services and checks are registered with an agent, which then
    takes on the burden of registering with the Catalog.

    Note, we don't actually have a centralized agent right now, since this is Redis not Consul.
    We have to manually manipulate all the checks ourselves.
    """

    def __init__(self, agent_name, kv):
        # type: (str, RedisKV) -> None
        self.kv = kv
        self.service = RedisService(agent_name, kv)
        self.check = RedisCheck(agent_name, kv)
        self.agent_name = agent_name
        self._auto_name = agent_name + "_autoservice"
        self._registered = False
        self._auto_register()

    def __del__(self):
        vprint("agent destructor: {}".format(self._auto_name))
        self.close()

    def _auto_register(self):
        self.service.register(self._auto_name)
        vprint("  registered:     {}".format(self._auto_name))
        self._registered = True

    def _auto_deregister(self):
        try:
            if self._registered:
                vprint("deregistered:     {}".format(self._auto_name))
                self.service.deregister(self._auto_name)
        except KeyError:
            pass  # ignore keyerror if something else cleaned it up
        self._registered = False

    def close(self):
        self._auto_deregister()

    def self(self):
        """
        Returns configuration of the local agent and member information.
        """
        return self.kv.info()

    def set_health(self, status):
        """Mark this node as healthy"""
        key = "health/checks/{}/{}".format(self.agent_name, self._auto_name)
        val = {"Status": status, "time": time.time()}
        return self.kv.put(key, val)

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

        Note, this excludes the envoy service itself in the response. 

        '/v1/agent/services'
        """
        return self.kv.get_dict("/catalog/services")

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
        return self.kv.client_list()

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


class RedisCatalog(Catalog):
    """This catalog is formed by aggregating information submitted by the agents. The catalog maintains the
    high-level view of the cluster, including which services are available, which nodes run those services,
    health information, and more. """

    def __init__(self, agent_name, kv):
        self.agent_name = agent_name
        self.kv = kv

    def register(self, node, address, service=None, check=None, dc=None, token=None):
        """
        A low level mechanism for directly registering or updating entries
        in the catalog. It is usually recommended to use
        agent.service.register and agent.check.register, as they are
        simpler and perform anti-entropy.

        '/v1/catalog/register'
        """
        raise NotImplementedError()

    def deregister(self, node, service_id=None, check_id=None, dc=None, token=None):
        """
        A low level mechanism for directly removing entries in the catalog.
        It is usually recommended to use the agent APIs, as they are
        simpler and perform anti-entropy.

        '/v1/catalog/deregister'
        """
        raise NotImplementedError()

    def datacenters(self):
        raise NotImplementedError("this api is single-datacenter only")

    def nodes(
        self, index=None, wait=None, consistency=None, dc=None, near=None, token=None, node_meta=None,
    ):
        """
        Returns a tuple of (*index*, *nodes*) of all nodes known
        about in the *dc* datacenter. *dc* defaults to the current
        datacenter of this agent.

        *index* is the current Consul index, suitable for making subsequent
        calls to wait for changes since this query was last run.
        Maps to /v1/catalog/nodes
        """

    def services(
        self, index=None, wait=None, consistency=None, dc=None, token=None, node_meta=None,
    ):
        """
        Returns a tuple of (*index*, *services*) of all services known
        about in the *dc* datacenter. *dc* defaults to the current
        datacenter of this agent.

        Maps to /v1/catalog/services
        """

    def node(self, node, index=None, wait=None, consistency=None, dc=None, token=None):
        """
        Returns a tuple of (*index*, *services*) of all services provided
        by *node*.
        """
        raise NotImplementedError()


class RedisService(Service):
    def __init__(self, agent_name, kv):
        # type: (str, RedisKV) -> None
        self.agent_name = agent_name
        self.kv = kv

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

        Example response from c.agent.services():
        'fooid': {'ID': 'fooid',
          'Service': 'fooname',
          'Tags': ['footag'],
          'Meta': {},
          'Port': 8500,
          'Address': 'localhost',
          'Weights': {'Passing': 1, 'Warning': 1},
          'EnableTagOverride': False}
        """
        if service_id is None:
            service_id = name
        key = "agent/{}/services/{}".format(self.agent_name, service_id)
        val = {
            "ID": service_id,
            "Service": name,
            "ttl": ttl,
        }

        return self.kv.put(key, val)

    def deregister(self, service_id):
        """
        Used to remove a service from the local agent. The agent will
        take care of deregistering the service with the Catalog. If
        there is an associated check, that is also deregistered.
        """
        key = "agent/{}/services/{}".format(self.agent_name, service_id)
        return self.kv.delete_dict(key)

    def maintenance(self, service_id, enable, reason=None):
        raise NotImplementedError("not used")


class RedisCheck(Check):
    def __init__(self, agent_name, kv):
        self.agent_name = agent_name
        self.kv = kv

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
        raise NotImplementedError("work in progress")

    def deregister(self, check_id):
        """
        Remove a check from the local agent.
        """
        raise NotImplementedError("work in progress")

    def ttl_pass(self, check_id, notes=None):
        """
        Mark a ttl based check as passing. Optional notes can be
        attached to describe the status of the check.
        """
        raise NotImplementedError("work in progress")

    def ttl_fail(self, check_id, notes=None):
        """
        Mark a ttl based check as failing. Optional notes can be
        attached to describe why check is failing. The status of the
        check will be set to critical and the ttl clock will be reset.
        """
        raise NotImplementedError("work in progress")

    def ttl_warn(self, check_id, notes=None):
        """
        Mark a ttl based check with warning. Optional notes can be
        attached to describe the warning. The status of the
        check will be set to warn and the ttl clock will be reset.
        """
        raise NotImplementedError("work in progress")


class RedisEnvoy(BaseEnvoy):
    """Redis-based KV and health checking API interface. Loosely shadows the python-consul interface
    KV functions are passed down to this class's namespace for convenience
    """

    __slots__ = ["heartbeat_timer"]

    def __init__(
        self,
        host="localhost",
        port=6379,
        db=0,
        password=None,
        socket_timeout=None,
        socket_connect_timeout=None,
        socket_keepalive=None,
        socket_keepalive_options=None,
        connection_pool=None,
        unix_socket_path=None,
        encoding="utf-8",
        encoding_errors="strict",
        charset=None,
        errors=None,
        decode_responses=False,
        retry_on_timeout=False,
        ssl=False,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_cert_reqs="required",
        ssl_ca_certs=None,
        ssl_check_hostname=False,
        max_connections=None,
        single_connection_client=False,
        health_check_interval=0,
        client_name=None,
        username=None,
    ):
        if client_name is None:
            from uuid import uuid4
            import socket

            client_name = socket.gethostname() + "_" + str(uuid4())[:8]

        cfgdict = dict(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=socket_keepalive,
            socket_keepalive_options=socket_keepalive_options,
            connection_pool=connection_pool,
            unix_socket_path=unix_socket_path,
            encoding=encoding,
            encoding_errors=encoding_errors,
            charset=charset,
            errors=errors,
            decode_responses=decode_responses,
            retry_on_timeout=retry_on_timeout,
            ssl=ssl,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_check_hostname=ssl_check_hostname,
            max_connections=max_connections,
            single_connection_client=single_connection_client,
            health_check_interval=health_check_interval,
            client_name=client_name,
            username=username,
        )

        self.client_name = client_name
        kv = RedisKV(client_name, cfgdict)
        self.kv = kv
        # self.txn = Consul.Txn(self)
        self.agent = RedisAgent(client_name, kv)
        self.catalog = RedisCatalog(client_name, kv)
        # self.session = Consul.Session(self)
        # self.acl = Consul.ACL(self)
        # self.status = Consul.Status(self)
        # self.query = Consul.Query(self)
        # self.coordinate = Consul.Coordinate(self)
        # self.operator = Consul.Operator(self)
        self.health = RedisHealth(client_name, kv)

    def __del__(self):
        """This destructor is a little weird and doesn't always trigger correctly"""
        vprint("envoy destructor: {}".format(self.name))
        self.close()

    def close(self):
        self.agent.close()
        self.kv.close()

    def get_dict(self, key, default=_default, flatten=False, **kwargs):
        return self.kv.get_dict(key, default, flatten=flatten, **kwargs)

    def get(self, key, default=_default, **kwargs):
        return self.kv.get(key, default, **kwargs)

    def put(self, key, val, **kwargs):
        return self.kv.put(key, val, **kwargs)

    def delete(self, key, **kwargs):
        return self.kv.delete(key, **kwargs)

    def delete_dict(self, key, **kwargs):
        return self.kv.delete_dict(key, **kwargs)

    def client_list(self):
        return self.kv.client_list()

    @property
    def name(self):
        return self.client_name


class StateService(ChildService):
    """A service that just posts some bit of state
    todo: merge with RedisService somehow.
    """

    def __init__(self, kv, node, name):
        #  type: (KV, str, str) -> None
        """
        :param kv: A key-value store object
        :param node: Name of node
        :param name: Name of service
        """
        self.node = node
        self.name = name
        super(StateService, self).__init__(kv=kv)

    def update_state(self, state, status="passing"):
        self.push(state)
        self.set_health(status=status)

    def set_health(self, status):
        """Set the current health status along with a bit of state. Use srv.push(state) to load before setting state"""
        key = "health/checks/{}/{}".format(self.node, self.name)
        val = {"Status": status, "time": time.time(), "state": self.peek()}
        return self.kv.put(key, val)


if __name__ == "__main__":
    import sys
    try:
        addr = sys.argv[1]
        host, port = addr.split(':')
    except IndexError as exc:
        print("Warning: {}".format(exc), file=sys.stderr)
        host = 'localhost'
        port = '6379'
    print("commence janky test on {}:{}".format(host, port), file=sys.stderr)
    dd = {"name": "bar_from_bd", "nest": {"spam": "eggs", "num": 42, "deep": {"a": 0}}, "a_list": [1, 2, "three"]}
    # rc = redis.Redis(host, port)
    kv = RedisKV('foo', {'host': host, 'port': port})
    print('keys at start: {}'.format(len(kv.keys())))
    key = '/TEST_DO_NOT_USE'
    kv.put(key, dd)
    out = kv.get(key)
    print('{}'.format(kv.keys(key + '*')))
    print(out)
    print(out == dd)
    kv.delete_dict(key)
    print('keys at end  : {}'.format(len(kv.keys())))
