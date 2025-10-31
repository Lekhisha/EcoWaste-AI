import { useState } from "react";
import { NavLink } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBars, faXmark } from "@fortawesome/free-solid-svg-icons";

const navLinks = [
  {
    id: 1,
    name: "Home",
    link: "/",
  },
  {
    id: 2,
    name: "How it works",
    link: "/howitworks",
  },
  {
    id: 3,
    name: "Scan or Upload",
    link: "/scanorupload",
  },
];

const Navbar = () => {
  {
    /* opening and closing toggle btn*/
  }
  const [isOpen, setisOpen] = useState(false);
  return (
    <>   
        <div className="fixed inset-x-0 top-0 z-50 bg-white/95 backdrop-blur-sm shadow-md ">
          <div className="container flex justify-between py-4 items-center">
            {/*logo*/}

            <div className=" text-3xl font-bold text-green-600">
              Eco<span className="text-cyan-950">-Waste AI</span>
            </div>

            {/*Navlinks*/}
            <div className="hidden md:flex  items-center gap-10 font-bold text-xl text-cyan-950 ">
              {navLinks.map(({ id, name, link }) => (
                <NavLink
                  key={id}
                  to={link}
                  className={({ isActive }) =>
                    isActive
                      ? "text-green-600 border-b-2"
                      : "text-cyan-950 hover:text-green-600"
                  }
                >
                  {name}
                </NavLink>
              ))}
            </div>

            <div className="block md:hidden ">
              <button onClick={() => setisOpen(!isOpen)}>
                {isOpen ? (
                  <FontAwesomeIcon icon={faXmark} size="lg" />
                ) : (
                  <FontAwesomeIcon icon={faBars} size="lg" />
                )}
              </button>
            </div>
          </div>
          {/*  toggle navbar */}
          {isOpen && (
            <div className="bg-green-500 w-full">
              <div className="flex flex-col items-center py-10 gap-6 text-white font-bold text-xl ">
                {navLinks.map(({ id, name, link }) => (
                  <NavLink
                    key={id}
                    to={link}
                    onClick={() => setisOpen(!isOpen)}
                    className={({ isActive }) =>
                      isActive
                        ? "text-cyan-950 border-b-2"
                        : " hover:text-cyan-950"
                    }
                  >
                    {name}
                  </NavLink>
                ))}
              </div>
            </div>
          )}
        </div>
      
    </>
  );
};

export default Navbar;
